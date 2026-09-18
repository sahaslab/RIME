import json
import random
import heapq
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any
from tqdm import tqdm
from plan_audit_utils import (
    load_configuration,
    load_analysis,
    extract_parameters,
    coverage_bins,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plans-path",
        type=Path,
        default=Path("derived/ground_truth/permissible_plans.jsonl"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("derived/ground_truth/subsampled_plans.jsonl"),
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("generation_priors/03_rime"),
    )
    parser.add_argument("--analysis-path", type=Path, default=None)
    parser.add_argument(
        "--policy",
        choices=["prior_coverage", "stratified_random", "random"],
        default="prior_coverage",
    )
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-per-source-recipe", type=int, default=3)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--input-limit", type=int, default=None)
    args = parser.parse_args()
    assert args.limit >= 0 and args.bins > 0 and args.max_per_source_recipe >= 0
    assert args.plans_path.resolve() != args.output_path.resolve()
    if args.policy == "stratified_random":
        tqdm.write(
            "stratified_random now uses prior_coverage; use random for uniform sampling."
        )
    routes, priors, _ = load_configuration(args.config_dir)
    analysis = load_analysis(args.analysis_path)
    candidates, features, unmodeled = read_candidates(args, routes, priors, analysis)
    selected, scores = select_candidates(candidates, features, args)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.plans_path.open() as source, args.output_path.open("w") as sink:
        for rank, index in enumerate(
            tqdm(selected, desc="Writing selected plans"),
            start=1,
        ):
            source.seek(candidates[index]["offset"])
            row = json.loads(source.readline())
            row.update(
                {
                    "selection_policy": "random"
                    if args.policy == "random"
                    else "prior_coverage",
                    "selection_rank": rank,
                    "selection_global_rank": rank,
                    "selection_score": scores[index],
                    "selection_seed": args.seed,
                    "selection_group": "global",
                }
            )
            sink.write(json.dumps(row, sort_keys=True) + "\n")
    available = Counter(feature for items in tqdm(features, desc="Summarizing coverage") for feature in items)
    covered = Counter(feature for index in selected for feature in features[index])
    report = {
        "requested": args.limit,
        "selected": len(selected),
        "candidates": len(candidates),
        "bins": args.bins,
        "config_dir": str(args.config_dir.resolve()),
        "analysis_path": str(args.analysis_path.resolve())
        if args.analysis_path
        else None,
        "max_per_source_recipe": args.max_per_source_recipe,
        "unmodeled_parameter_occurrences": dict(unmodeled),
        "coverage": [
            {
                "dimension": dimension,
                "bin": bucket,
                "available_plans": count,
                "selected_plans": covered[(dimension, bucket)],
            }
            for (dimension, bucket), count in sorted(available.items())
        ],
    }
    report_path = args.output_path.with_suffix(".coverage.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    tqdm.write(
        "Selected %d/%d requested plans; covered %d/%d available features. Report: %s"
        % (len(selected), args.limit, len(covered), len(available), report_path)
    )
    if unmodeled:
        tqdm.write(
            "Some parameters lack a resolved prior; see unmodeled_parameter_occurrences in the report."
        )


def read_candidates(
    args: argparse.Namespace,
    routes: dict[str, Any],
    priors: dict[str, Any],
    analysis: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[set[tuple[str, str]]], Counter]:
    candidates, features = [], []
    pending = defaultdict(list)
    models = {}
    unmodeled = Counter()
    with (
        args.plans_path.open() as handle,
        tqdm(desc="Reading candidate parameters", unit="plan") as progress,
    ):
        while args.input_limit is None or len(candidates) < args.input_limit:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            index = len(candidates)
            candidates.append(
                {
                    "offset": offset,
                    "clip_id": row["clip_id"],
                    "recipe_id": row["recipe_id"],
                }
            )
            features.append(
                {
                    ("recipe", row["recipe_id"]),
                    ("target_family", str(row.get("target_family") or "unspecified")),
                }
            )
            values, missing = extract_parameters(
                row, routes, priors, row.get("analysis", analysis.get(row["clip_id"]))
            )
            unmodeled.update(item["reason"] for item in missing)
            for item in values:
                model = item["model"]
                if model is None:
                    unmodeled[item["status"]] += 1
                elif item["status"] in {"prior", "authored"}:
                    name = item["prior"]
                    models[name] = model
                    pending[name].append((index, item["value"]))
            progress.update(1)
    for name, occurrences in tqdm(pending.items(), desc="Binning prior values"):
        bins = coverage_bins(
            models[name], [value for _, value in occurrences], args.bins
        )
        for (index, _), bucket in tqdm(
            zip(occurrences, bins), total=len(occurrences), desc=name, leave=False
        ):
            if bucket is not None:
                features[index].add((name, bucket))
            else:
                unmodeled["Outside prior support"] += 1
    return candidates, features, unmodeled


def select_candidates(
    candidates: list[dict[str, Any]],
    features: list[set[tuple[str, str]]],
    args: argparse.Namespace,
) -> tuple[list[int], dict[int, float]]:
    order = list(range(len(candidates)))
    random.Random(args.seed).shuffle(order)
    counts = Counter()
    source_counts = Counter()
    selected = []
    scores = {}
    if args.policy == "random":
        for index in tqdm(order, desc="Selecting random plans"):
            if len(selected) >= args.limit:
                break
            candidate = candidates[index]
            key = (candidate["clip_id"], candidate["recipe_id"])
            if (
                args.max_per_source_recipe
                and source_counts[key] >= args.max_per_source_recipe
            ):
                continue
            selected.append(index)
            scores[index] = 0.0
            source_counts[key] += 1
        return selected, scores

    # Each prior has total feature weight one, regardless of its number of bins.
    dimensions = defaultdict(set)
    for items in tqdm(features, desc="Counting available bins"):
        for dimension, bucket in items:
            dimensions[dimension].add(bucket)
    weights = {
        dimension: 1.0 / len(buckets) for dimension, buckets in dimensions.items()
    }
    groups = defaultdict(list)
    for index in tqdm(order, desc="Grouping recipes and targets"):
        for feature in features[index]:
            if feature[0] in {"recipe", "target_family"}:
                groups[feature].append(index)
    for feature in tqdm(
        sorted(groups, key=lambda key: (len(groups[key]), key)),
        desc="Covering recipes and targets",
    ):
        if len(selected) >= args.limit:
            break
        if counts[feature]:
            continue
        eligible = [
            index
            for index in groups[feature]
            if not args.max_per_source_recipe
            or source_counts[
                (candidates[index]["clip_id"], candidates[index]["recipe_id"])
            ]
            < args.max_per_source_recipe
        ]
        if not eligible:
            continue
        index = max(
            eligible, key=lambda item: marginal_gain(features[item], counts, weights)
        )
        selected.append(index)
        scores[index] = marginal_gain(features[index], counts, weights)
        counts.update(features[index])
        source_counts[
            (candidates[index]["clip_id"], candidates[index]["recipe_id"])
        ] += 1
    heap = [
        (-marginal_gain(features[index], counts, weights), tie, index)
        for tie, index in enumerate(tqdm(order, desc="Ranking coverage candidates"))
        if index not in scores
    ]
    heapq.heapify(heap)
    # Diminishing gains make cached scores upper bounds, enabling lazy greedy selection.
    with tqdm(
        total=min(args.limit, len(candidates)),
        initial=len(selected),
        desc="Selecting coverage plans",
    ) as progress:
        while heap and len(selected) < args.limit:
            _, tie, index = heapq.heappop(heap)
            candidate = candidates[index]
            key = (candidate["clip_id"], candidate["recipe_id"])
            if (
                args.max_per_source_recipe
                and source_counts[key] >= args.max_per_source_recipe
            ):
                continue
            score = marginal_gain(features[index], counts, weights)
            entry = (-score, tie, index)
            if heap and entry > heap[0]:
                heapq.heappush(heap, entry)
                continue
            selected.append(index)
            scores[index] = score
            counts.update(features[index])
            source_counts[key] += 1
            progress.update(1)
    return selected, scores


def marginal_gain(
    features: set[tuple[str, str]],
    counts: Counter,
    weights: dict[str, float],
) -> float:
    return sum(
        weights[dimension] / (1 + counts[(dimension, bucket)])
        for dimension, bucket in sorted(features)
    )


if __name__ == "__main__":
    main()
