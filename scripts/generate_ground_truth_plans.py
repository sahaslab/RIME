import sys
import hashlib
import argparse
import multiprocessing
from pathlib import Path
from collections.abc import Mapping, Iterator
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_truth.planner import GroundTruthPlanner
from ground_truth.io_utils import write_jsonl, load_records
from typing import Any
from tqdm.auto import tqdm


WORKER_PLANNER: GroundTruthPlanner | None = None
WORKER_MAX_VARIANTS_PER_RECIPE: int | None = None
WORKER_VARIANT_SELECTION: str = "diverse"
WORKER_RANDOM_PLANS_PER_CLIP: int = 8
WORKER_SEED: int = 0
WORKER_POISON_ONLY: bool = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-path", type=Path, default=Path("~/lab/postmaster/ground_truth/mtg_jamendo_analysis_manifest.jsonl").expanduser(), help="Input analysis manifest. Default: %(default)s")
    parser.add_argument("--output-path", type=Path, default=Path("~/lab/postmaster/ground_truth/permissible_plans.jsonl"), help="Output JSONL path for symbolic permissible plans. Default: %(default)s")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/ground_truth"), help="Ground-truth config directory. Default: %(default)s")
    parser.add_argument("--max-variants-per-recipe", type=int, default=None, help="Optional hard cap per recipe per clip. Default: no cap")
    parser.add_argument(
        "--variant-selection",
        choices=["diverse", "random"],
        default="random",
        help="How to choose variants when --max-variants-per-recipe is set. Default: %(default)s",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for pseudo-random per-clip variant selection. Default: %(default)s",
    )
    parser.add_argument("--random-plans-per-clip", type=int, default=8, help="Additional constrained random plans per clip. Default: %(default)s")
    parser.add_argument("--disable-random-plans", action="store_true", help="Disable constrained random plan generation")
    parser.add_argument("--poison-only", action="store_true", help="Keep only plans that include a poison graph.")
    parser.add_argument("--limit", type=int, default=None, help="Optional clip limit for smoke tests or partial generation. Default: no limit")
    parser.add_argument("--num-workers", type=int, default=None, help="Optional number of worker processes for parallel processing. Default: no limit")
    args = parser.parse_args()

    records = load_records(args.analysis_path)
    if args.limit is not None:
        records = records[:args.limit]

    num_workers = args.num_workers or multiprocessing.cpu_count()
    chunksize = _pool_chunksize(num_workers)

    write_jsonl(
        args.output_path,
        iter_rows(
            records=records,
            config_dir=args.config_dir,
            max_variants_per_recipe=args.max_variants_per_recipe,
            variant_selection=args.variant_selection,
            random_plans_per_clip=0 if (args.disable_random_plans or args.poison_only) else args.random_plans_per_clip,
            seed=args.seed,
            poison_only=bool(args.poison_only),
            num_workers=num_workers,
            chunksize=chunksize
        )
    )


def iter_rows(
    records: list[dict[str, Any]],
    config_dir: Path,
    max_variants_per_recipe: int | None,
    variant_selection: str,
    random_plans_per_clip: int,
    seed: int,
    poison_only: bool,
    num_workers: int,
    chunksize: int
) -> Iterator[dict[str, Any]]:
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(config_dir, max_variants_per_recipe, variant_selection, random_plans_per_clip, seed, poison_only),
    ) as pool:
        for rows_batch in tqdm(pool.imap(build_rows, records, chunksize=chunksize), total=len(records), desc="Building rows"):
            yield from rows_batch


def init_worker(
    config_dir: Path,
    max_variants_per_recipe: int | None,
    variant_selection: str,
    random_plans_per_clip: int,
    seed: int,
    poison_only: bool,
) -> None:
    global WORKER_PLANNER, WORKER_MAX_VARIANTS_PER_RECIPE, WORKER_VARIANT_SELECTION, WORKER_RANDOM_PLANS_PER_CLIP, WORKER_SEED, WORKER_POISON_ONLY
    WORKER_PLANNER = GroundTruthPlanner.from_directory(config_dir)
    WORKER_MAX_VARIANTS_PER_RECIPE = max_variants_per_recipe
    WORKER_VARIANT_SELECTION = variant_selection
    WORKER_RANDOM_PLANS_PER_CLIP = random_plans_per_clip
    WORKER_SEED = seed
    WORKER_POISON_ONLY = poison_only


def build_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if WORKER_PLANNER is None:
        raise RuntimeError("Worker planner was not initialized.")

    plan_seed = stable_seed(record.get("clip_id"), record.get("audio_path"), WORKER_SEED)
    plans = WORKER_PLANNER.plan(
        metadata=record,
        mode="enumerate",
        max_variants_per_recipe=WORKER_MAX_VARIANTS_PER_RECIPE,
        variant_selection=WORKER_VARIANT_SELECTION,
        seed=plan_seed
    )
    plans.extend(
        WORKER_PLANNER.random_plans(
            metadata=record,
            count=WORKER_RANDOM_PLANS_PER_CLIP,
            seed=stable_seed(record.get("clip_id"), record.get("audio_path"), "random", WORKER_SEED),
        )
    )
    analysis = dict(record.get("analysis", {}))
    genres = list(analysis.get("genres", []))
    mood_themes = list(analysis.get("mood_themes", []))
    issues = list(analysis.get("issues", []))
    rows: list[dict[str, Any]] = []
    for plan in plans:
        if WORKER_POISON_ONLY and plan.poison_graph_spec is None:
            continue
        row = plan.to_dict()
        row["clip_id"] = record.get("clip_id")
        row["audio_path"] = record.get("audio_path")
        row["genres"] = genres
        row["mood_themes"] = mood_themes
        row["issues"] = issues
        row["target_stem"] = plan.bindings.get("target_description")
        row["target_family"] = plan.bindings.get("target_family")
        target_candidate = plan.bindings.get("target_candidate") or {}
        row["separation_target"] = target_candidate.get("separation_target")
        row["graph_description"] = WORKER_PLANNER.describe_plan(plan, validate=False)
        row["poison_graph_description"] = (
            None
            if plan.poison_graph_spec is None
            else WORKER_PLANNER.describe_graph_spec(plan.poison_graph_spec, validate=False)
        )
        rows.append(row)
    return rows


def _pool_chunksize(num_workers: int) -> int:
    # Small batches preserve load balance because per-record expansion is highly skewed
    return 1 if num_workers <= 4 else 2


def stable_seed(*parts: object) -> int:
    joined = "||".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


if __name__ == "__main__":
    main()
