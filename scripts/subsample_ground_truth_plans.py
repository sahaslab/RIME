import json
import random
import hashlib
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import numpy as np
from typing import Any
from tqdm.auto import tqdm


@dataclass
class Candidate:
    candidate_id: str
    clip_id: str
    recipe_id: str
    plan_id: str
    row: dict[str, Any]
    joint_text: str
    selection_score: float = 0.0


def stage(message: str) -> None:
    tqdm.write(f"[subsample] {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plans-path",
        type=Path,
        default=Path("~/lab/postmaster/ground_truth/permissible_plans.jsonl").expanduser(),
        help="Input JSONL of permissible plans. Default: %(default)s",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("~/lab/postmaster/ground_truth/subsampled_plans.jsonl").expanduser(),
        help="Output JSONL for selected plans. Default: %(default)s",
    )
    parser.add_argument(
        "--policy",
        choices=["stratified_random", "stratified_feature_submodular", "feature_submodular", "random"],
        default="stratified_feature_submodular",
        help="Global selection policy. Default: %(default)s",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Number of plans to select globally. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed. Default: %(default)s",
    )
    parser.add_argument(
        "--max-per-source-recipe",
        type=int,
        default=3,
        help="Maximum selected plans per clip/source and recipe. Default: %(default)s",
    )
    parser.add_argument(
        "--input-limit",
        type=int,
        default=None,
        help="Optional limit on input rows for smoke tests. Default: no limit",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model used for joint plan embeddings. Default: %(default)s",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Embedding batch size. Default: %(default)s",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional sentence-transformer device override, e.g. cpu or cuda. Default: model default",
    )
    parser.add_argument(
        "--optimizer",
        choices=["stochastic", "sample", "approximate-lazy", "two-stage", "lazy", "naive"],
        default="stochastic",
        help="Apricot optimizer. Default: %(default)s",
    )
    parser.add_argument(
        "--optimizer-epsilon",
        type=float,
        default=0.9,
        help="Epsilon for apricot stochastic/sample optimizers. Default: %(default)s",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional cache directory. Default: <output-path>.cache",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable reading/writing caches.",
    )
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir is not None
        else Path(str(args.output_path) + ".cache").expanduser().resolve()
    )
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    stage(f"loading candidates from {args.plans_path}")
    candidates = load_candidates(args.plans_path, args.input_limit)
    stage(f"loaded {len(candidates)} candidates")
    if not candidates:
        write_selected_rows(
            output_path=args.output_path,
            selected=[],
            policy=args.policy,
            seed=args.seed,
            embedding_model=args.embedding_model,
        )
        return

    target = min(args.limit, len(candidates))
    stage(f"target selection count: {target}")

    if args.policy == "random":
        stage("running random baseline")
        selected = random_select(candidates, target, args.seed)
    elif args.policy == "stratified_random":
        stage("running stratified random baseline")
        ranking = random_ranking(candidates, args.seed)
        selected = select_stratified_from_ranking(
            candidates,
            ranking,
            target,
            args.seed,
            args.max_per_source_recipe,
        )
    else:
        stage(f"building/loading joint embedding features in {cache_dir}")
        features = build_feature_matrix(
            candidates=candidates,
            embedding_model=args.embedding_model,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=cache_dir,
            use_cache=not args.no_cache,
            plans_path=args.plans_path,
            input_limit=args.input_limit,
        )
        stage(f"feature matrix shape={features.shape}")
        stage(f"running apricot feature selection with optimizer={args.optimizer}")
        ranking = rank_with_apricot(
            features=features,
            n_samples=target,
            optimizer=args.optimizer,
            optimizer_epsilon=args.optimizer_epsilon,
            seed=args.seed,
            cache_dir=cache_dir,
            use_cache=not args.no_cache,
            plans_path=args.plans_path,
            input_limit=args.input_limit,
            limit=target,
            embedding_model=args.embedding_model,
        )
        if args.policy == "stratified_feature_submodular":
            selected = select_stratified_from_ranking(
                candidates,
                ranking,
                target,
                args.seed,
                args.max_per_source_recipe,
            )
        else:
            selected = select_from_ranking(candidates, ranking, target)

    stage(f"writing {len(selected)} selected rows to {args.output_path}")
    write_selected_rows(
        output_path=args.output_path,
        selected=selected,
        policy=args.policy,
        seed=args.seed,
        embedding_model=args.embedding_model,
    )
    report_selected_stats(selected)
    stage("done")


def load_candidates(path: Path, input_limit: int | None) -> list[Candidate]:
    candidates: list[Candidate] = []
    with path.open("r", encoding="utf-8") as handle:
        progress = tqdm(desc="Loading plans", unit="plan")
        for index, line in enumerate(handle):
            if input_limit is not None and index >= input_limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            clip_id = str(row.get("clip_id"))
            plan_id = str(row.get("plan_id"))
            recipe_id = str(row.get("recipe_id"))
            candidates.append(
                Candidate(
                    candidate_id=f"{clip_id}::{plan_id}::{index}",
                    clip_id=clip_id,
                    recipe_id=recipe_id,
                    plan_id=plan_id,
                    row=row,
                    joint_text=build_joint_text(row),
                )
            )
            progress.update(1)
        progress.close()
    return candidates


def build_joint_text(row: Mapping[str, Any]) -> str:
    parts: list[str] = []
    if row.get("clip_id"):
        parts.append(f"clip: {row.get('clip_id')}")
    if row.get("genres"):
        parts.append("genres: " + ", ".join(str(value) for value in row.get("genres") or []))
    if row.get("mood_themes"):
        parts.append("moods: " + ", ".join(str(value) for value in row.get("mood_themes") or []))
    if row.get("issues"):
        parts.append("issues: " + ", ".join(str(value) for value in row.get("issues") or []))
    if row.get("target_stem"):
        parts.append(f"target stem: {row.get('target_stem')}")
    if row.get("target_family"):
        parts.append(f"target family: {row.get('target_family')}")
    bindings = row.get("bindings") or {}
    candidate = bindings.get("target_candidate") or {}
    if candidate:
        parts.append("target candidate: " + ", ".join(f"{k}={candidate[k]}" for k in sorted(candidate)))
    if row.get("recipe_id"):
        parts.append(f"recipe: {row.get('recipe_id')}")
    if row.get("recipe_tags"):
        parts.append("recipe tags: " + ", ".join(str(tag) for tag in row.get("recipe_tags") or []))
    if row.get("applied_policies"):
        parts.append("policies: " + ", ".join(str(policy) for policy in row.get("applied_policies") or []))
    graph_description = row.get("graph_description")
    if graph_description:
        parts.append(str(graph_description).replace("\n", " | "))
    else:
        parts.append(render_graph_spec_text(row.get("graph_spec") or []))
    return " | ".join(part for part in parts if part)


def render_graph_spec_text(graph_spec: Sequence[Mapping[str, Any]]) -> str:
    parts: list[str] = []
    for block in graph_spec:
        kind = block.get("kind")
        if kind == "separate":
            parts.append(f"separate {block.get('description')} -> {block.get('outputs')}")
        elif kind == "mix":
            parts.append(f"mix stem={block.get('stem')} residual={block.get('residual')} -> {block.get('output')}")
        elif kind in {"chain", "send_return"}:
            steps = []
            for step in block.get("steps", []):
                params = step.get("params") or {}
                if params:
                    rendered = ", ".join(f"{k}={params[k]}" for k in sorted(params))
                    steps.append(f"{step.get('operator')}({rendered})")
                else:
                    steps.append(str(step.get("operator")))
            parts.append(f"{kind} {block.get('source')} -> {' -> '.join(steps)} -> {block.get('output')}")
        elif kind == "step":
            params = block.get("params") or {}
            if params:
                rendered = ", ".join(f"{k}={params[k]}" for k in sorted(params))
                parts.append(f"step {block.get('operator')}({rendered})")
            else:
                parts.append(f"step {block.get('operator')}")
    return " | ".join(parts)


def random_select(candidates: Sequence[Candidate], limit: int, seed: int) -> list[Candidate]:
    ranking = random_ranking(candidates, seed)
    selected = [candidates[index] for index in ranking[:limit]]
    for rank, candidate in enumerate(selected, start=1):
        candidate.selection_score = float(limit - rank)
    return selected


def random_ranking(candidates: Sequence[Candidate], seed: int) -> list[int]:
    ranking = list(range(len(candidates)))
    random.Random(seed).shuffle(ranking)
    random_positions = {
        index: position
        for position, index in enumerate(ranking)
    }
    ranking.sort(
        key=lambda index: (
            -target_family_priority(candidates[index].row),
            random_positions[index],
        )
    )
    return ranking


def cache_key(prefix: str, **parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def build_feature_matrix(
    *,
    candidates: Sequence[Candidate],
    embedding_model: str,
    batch_size: int,
    device: str | None,
    cache_dir: Path,
    use_cache: bool,
    plans_path: Path,
    input_limit: int | None,
) -> np.ndarray:
    key = cache_key(
        "features",
        plans_path=str(plans_path.resolve()),
        plans_mtime=plans_path.stat().st_mtime_ns,
        plans_size=plans_path.stat().st_size,
        input_limit=input_limit,
        embedding_model=embedding_model,
        candidate_ids=[candidate.candidate_id for candidate in candidates],
    )
    cache_path = cache_dir / f"{key}.npz"
    if use_cache and cache_path.exists():
        stage(f"loading cached feature matrix from {cache_path}")
        return np.load(cache_path)["features"]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(embedding_model, device=device) if device else SentenceTransformer(embedding_model)
    unique_texts = sorted({candidate.joint_text for candidate in candidates})
    stage(f"encoding {len(unique_texts)} unique datapoint texts")
    embeddings = encode_texts(model, unique_texts, batch_size)
    embedding_map = {text: embedding for text, embedding in zip(unique_texts, embeddings)}

    stage("assembling positive feature matrix")
    rows: list[np.ndarray] = []
    for candidate in tqdm(candidates, desc="Assembling features", unit="plan"):
        row = embedding_map[candidate.joint_text].copy()
        row = 0.5 * (row + 1.0)
        rows.append(row.astype(np.float32, copy=False))
    features = np.vstack(rows)

    if use_cache:
        np.savez_compressed(cache_path, features=features)
        stage(f"saved feature cache to {cache_path}")
    return features


def encode_texts(model: Any, texts: Sequence[str], batch_size: int) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def rank_with_apricot(
    *,
    features: np.ndarray,
    n_samples: int,
    optimizer: str,
    optimizer_epsilon: float,
    seed: int,
    cache_dir: Path,
    use_cache: bool,
    plans_path: Path,
    input_limit: int | None,
    limit: int,
    embedding_model: str,
) -> list[int]:
    key = cache_key(
        "ranking",
        plans_path=str(plans_path.resolve()),
        plans_mtime=plans_path.stat().st_mtime_ns,
        plans_size=plans_path.stat().st_size,
        input_limit=input_limit,
        feature_shape=tuple(features.shape),
        feature_sum=float(features.sum()),
        limit=limit,
        n_samples=n_samples,
        optimizer=optimizer,
        optimizer_epsilon=optimizer_epsilon,
        seed=seed,
        embedding_model=embedding_model,
    )
    cache_path = cache_dir / f"{key}.json"
    if use_cache and cache_path.exists():
        stage(f"loading cached ranking from {cache_path}")
        return json.loads(cache_path.read_text())

    import apricot

    optimizer_kwds: dict[str, Any] = {}
    if optimizer in {"stochastic", "sample"}:
        optimizer_kwds["epsilon"] = optimizer_epsilon
    selector = apricot.FeatureBasedSelection(
        n_samples=n_samples,
        concave_func="sqrt",
        optimizer=optimizer,
        optimizer_kwds=optimizer_kwds,
        random_state=seed,
        verbose=True,
    )
    selector.fit(features)
    ranking = [int(index) for index in selector.ranking]
    if use_cache:
        cache_path.write_text(json.dumps(ranking))
        stage(f"saved ranking cache to {cache_path}")
    return ranking


def select_from_ranking(candidates: Sequence[Candidate], ranking: Sequence[int], limit: int) -> list[Candidate]:
    selected: list[Candidate] = []
    for rank, index in enumerate(tqdm(ranking[:limit], desc="Selecting ranked plans", unit="plan"), start=1):
        candidate = candidates[int(index)]
        candidate.selection_score = float(limit - rank)
        selected.append(candidate)
    return selected


def select_stratified_from_ranking(
    candidates: Sequence[Candidate],
    ranking: Sequence[int],
    limit: int,
    seed: int,
    max_per_source_recipe: int,
) -> list[Candidate]:
    ranked_candidates = [candidates[int(index)] for index in ranking]
    family_bins: dict[str, dict[str, list[Candidate]]] = {}
    for candidate in ranked_candidates:
        family = effect_family(candidate.row)
        param_bin = param_bin_signature(candidate.row)
        family_bins.setdefault(family, {})
        family_bins[family].setdefault(param_bin, [])
        family_bins[family][param_bin].append(candidate)

    rng = random.Random(seed)
    family_names = sorted(family_bins)
    rng.shuffle(family_names)
    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    source_recipe_counts: Counter[tuple[str, str]] = Counter()
    source_recipe_identity_counts: Counter[tuple[str, str, str]] = Counter()
    while len(selected) < limit and family_names:
        progressed = False
        for family in list(family_names):
            bin_names = sorted(family_bins[family])
            rng.shuffle(bin_names)
            picked = None
            for bin_name in bin_names:
                queue = family_bins[family][bin_name]
                while queue:
                    candidate = queue.pop(0)
                    if candidate.candidate_id in selected_ids:
                        continue
                    source_recipe_key = (candidate.clip_id, candidate.recipe_id)
                    source_recipe_identity_key = (candidate.clip_id, candidate.recipe_id, bin_name)
                    if max_per_source_recipe > 0 and source_recipe_counts[source_recipe_key] >= max_per_source_recipe:
                        selected_ids.add(candidate.candidate_id)
                        continue
                    if source_recipe_identity_counts[source_recipe_identity_key] >= 1:
                        selected_ids.add(candidate.candidate_id)
                        continue
                    picked = candidate
                    break
            if picked is None:
                family_names.remove(family)
                continue
            picked.selection_score = float(limit - len(selected))
            selected.append(picked)
            selected_ids.add(picked.candidate_id)
            source_recipe_key = (picked.clip_id, picked.recipe_id)
            source_recipe_identity_key = (picked.clip_id, picked.recipe_id, param_bin_signature(picked.row))
            source_recipe_counts[source_recipe_key] += 1
            source_recipe_identity_counts[source_recipe_identity_key] += 1
            progressed = True
            if len(selected) >= limit:
                break
        if not progressed:
            break
    return selected


def effect_family(row: Mapping[str, Any]) -> str:
    tags = set(str(tag) for tag in row.get("recipe_tags") or [])
    operators = set(operator_names(row.get("graph_spec") or []))
    if "harmony" in tags or "apply_harmony_effect" in operators:
        return "harmony"
    if "delay" in tags or "apply_delay_effect" in operators:
        if "reverb" in tags or "apply_reverb_effect" in operators:
            return "delay_reverb"
        return "delay"
    if "reverb" in tags or "apply_reverb_effect" in operators:
        return "reverb"
    if "compression" in tags or "apply_compressor_effect" in operators:
        return "compression"
    if "gain" in tags or "apply_gain" in operators:
        return "gain"
    if "modulation" in tags or "apply_chorus_effect" in operators or "apply_phaser_effect" in operators:
        return "modulation"
    if "eq" in tags:
        return "eq"
    return "other"


def operator_names(graph_spec: Sequence[Mapping[str, Any]]) -> list[str]:
    names: list[str] = []
    for block in graph_spec:
        if block.get("operator"):
            names.append(str(block.get("operator")))
        for step in block.get("steps", []):
            if step.get("operator"):
                names.append(str(step.get("operator")))
    return names


def param_bin_signature(row: Mapping[str, Any]) -> str:
    if row.get("recipe_id") == "vocal_harmony_support":
        return "harmony_interval=%s" % harmony_interval(row)
    tags = set(str(tag) for tag in row.get("recipe_tags") or [])
    if "delay" in tags and "reverb" in tags:
        return "delay=%s|reverb_room=%s" % (delay_identity(row), reverb_identity(row))
    if "delay" in tags:
        return "delay=%s" % delay_identity(row)
    if "reverb" in tags:
        return "reverb_room=%s" % reverb_identity(row)
    bins: list[str] = []
    for block in row.get("graph_spec") or []:
        for step in block.get("steps", []):
            operator = str(step.get("operator"))
            for name, value in sorted((step.get("params") or {}).items()):
                bins.append("%s.%s=%s" % (operator, name, numeric_bin(name, value)))
        if block.get("kind") == "send_return":
            for name in ["send_level", "return_level", "dry_level"]:
                if name in block:
                    bins.append("send_return.%s=%s" % (name, numeric_bin(name, block[name])))
    return "|".join(bins[:6]) if bins else "no_params"


def target_family_priority(row: Mapping[str, Any]) -> float:
    family = str(row.get("target_family") or "")
    priorities = {
        "vocals": 4.0,
        "drums": 3.0,
        "guitar": 1.5,
        "bass": 1.25,
        "piano": 1.0,
    }
    return priorities.get(family, 0.25)


def harmony_interval(row: Mapping[str, Any]) -> str:
    for operator, name, value in iter_operator_params(row.get("graph_spec") or []):
        if operator == "apply_harmony_effect" and name == "semitones":
            return numeric_bin(name, value)
    return "unknown"


def delay_identity(row: Mapping[str, Any]) -> str:
    for operator, name, value in iter_operator_params(row.get("graph_spec") or []):
        if operator == "apply_delay_effect" and name == "delay_seconds":
            return numeric_bin(name, value)
    return "unknown"


def reverb_identity(row: Mapping[str, Any]) -> str:
    for operator, name, value in iter_operator_params(row.get("graph_spec") or []):
        if operator == "apply_reverb_effect" and name == "room_size":
            return numeric_bin(name, value)
    return "unknown"


def iter_operator_params(graph_spec: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, Any]]:
    params: list[tuple[str, str, Any]] = []
    for block in graph_spec:
        for step in block.get("steps", []):
            operator = str(step.get("operator"))
            for name, value in sorted((step.get("params") or {}).items()):
                params.append((operator, str(name), value))
        if block.get("operator"):
            operator = str(block.get("operator"))
            for name, value in sorted((block.get("params") or {}).items()):
                params.append((operator, str(name), value))
    return params


def numeric_bin(name: str, value: Any) -> str:
    if not isinstance(value, (int, float)):
        return str(value)
    numeric = float(value)
    if name in {"mix", "wet_level", "send_level", "return_level"}:
        if numeric < 0.34:
            return "low"
        if numeric < 0.67:
            return "mid"
        return "high"
    if name in {"delay_seconds"}:
        if numeric < 0.18:
            return "slap"
        if numeric < 0.32:
            return "short"
        if numeric < 0.55:
            return "medium"
        return "long"
    if name in {"room_size"}:
        if numeric < 0.38:
            return "small"
        if numeric < 0.63:
            return "medium"
        if numeric < 0.85:
            return "large"
        return "huge"
    if name in {"semitones"}:
        return "%+d" % int(round(numeric))
    if name in {"feedback"}:
        if numeric <= 0.01:
            return "none"
        if numeric < 0.25:
            return "low"
        return "high"
    if name in {"gain_db", "drive_db"}:
        if abs(numeric) < 3.0:
            return "subtle"
        if abs(numeric) < 6.0:
            return "clear"
        return "strong"
    return "%.3g" % numeric


def write_selected_rows(
    *,
    output_path: Path,
    selected: Sequence[Candidate],
    policy: str,
    seed: int,
    embedding_model: str,
) -> None:
    with output_path.open("w", encoding="utf-8") as sink:
        progress = tqdm(total=len(selected), desc="Writing selected rows", unit="plan")
        for rank, candidate in enumerate(selected, start=1):
            updated = dict(candidate.row)
            updated["selection_policy"] = policy
            updated["selection_group"] = "global"
            updated["selection_rank"] = rank
            updated["selection_global_rank"] = rank
            updated["selection_score"] = candidate.selection_score
            updated["selection_seed"] = seed
            updated["selection_candidate_id"] = candidate.candidate_id
            updated["selection_clip_id"] = candidate.clip_id
            updated["selection_embedding_model"] = embedding_model
            sink.write(json.dumps(updated, sort_keys=True))
            sink.write("\n")
            progress.update(1)
        progress.close()


def report_selected_stats(selected: Sequence[Candidate]) -> None:
    clip_counts = Counter(candidate.clip_id for candidate in selected)
    recipe_counts = Counter(candidate.recipe_id for candidate in selected)
    clip_recipe_counts = Counter((candidate.clip_id, candidate.recipe_id) for candidate in selected)
    stage(f"posthoc stats | max_per_clip={max(clip_counts.values(), default=0)} | max_per_recipe={max(recipe_counts.values(), default=0)} | max_per_clip_recipe={max(clip_recipe_counts.values(), default=0)}")
    stage(f"top recipes: {recipe_counts.most_common(10)}")
    stage(f"top clips: {clip_counts.most_common(10)}")


if __name__ == "__main__":
    main()
