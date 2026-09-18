import sys
import json
import argparse
import hashlib
from pathlib import Path
from tqdm import tqdm


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "derived/ground_truth/musiccaps_analysis_manifest.jsonl")
    parser.add_argument("--config-dir", type=Path, default=root / "generation_priors/03_rime")
    parser.add_argument("--rime-root", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "derived/ground_truth/prior_draws.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--samples-per-recipe", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    assert args.limit > 0 and args.samples_per_recipe > 0
    sys.path.insert(0, str(args.rime_root))
    from ground_truth.planner import GroundTruthPlanner
    planner = GroundTruthPlanner.from_directory(args.config_dir)
    records = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()][:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w") as handle:
        for record in tqdm(records, desc="Sampling plans for audit"):
            identity = json.dumps([args.seed, record["clip_id"]])
            seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
            plans = planner.plan(record, mode="sample", samples_per_recipe=args.samples_per_recipe, seed=seed)
            for plan in plans:
                handle.write(json.dumps(plan.to_dict() | {"clip_id": record["clip_id"], "analysis": record["analysis"], "generation_mode": "sample", "generation_seed": seed}, allow_nan=False) + "\n")
                count += 1
    paths = [args.manifest] + sorted(args.config_dir.glob("*.yaml"))
    metadata = {"mode": "sample", "clips": len(records), "plans": count, "samples_per_recipe": args.samples_per_recipe, "seed": args.seed, "subsampled": False, "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    args.output.with_suffix(".settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Sampled %d plans from %d clips: %s" % (count, len(records), args.output))


if __name__ == "__main__":
    main()
