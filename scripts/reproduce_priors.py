import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path
from tqdm import tqdm


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-path", type=Path, default=root / "data/ground_truth/musiccaps_analysis_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=root / "derived/ground_truth")
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-random-fraction", type=float, default=0.15)
    parser.add_argument("--max-poison-fraction", type=float, default=0.0)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--start-at", choices=["corpus", "fit", "export", "verify", "generate", "sample", "audit"], default="corpus")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = root / "generation_priors/03_rime"
    plans = args.output_dir / "permissible_plans.jsonl"
    selected = args.output_dir / "subsampled_plans.jsonl"
    jobs = [
        ("corpus", "build_corpus.py", ["--offline"] if args.offline else []),
        ("fit", "extract_priors.py", []),
        ("export", "export_priors_to_yaml.py", []),
        ("verify", "verify_sampling.py", ["--analysis-path", str(args.analysis_path), "--output-path", str(args.output_dir / "sampling_checks.json")]),
        ("generate", "generate_ground_truth_plans.py", ["--analysis-path", str(args.analysis_path), "--config-dir", str(config), "--output-path", str(plans), "--sampling-mode", "sample", "--seed", str(args.seed), "--num-workers", str(args.workers)]),
        ("sample", "subsample_ground_truth_plans.py", ["--plans-path", str(plans), "--output-path", str(selected), "--config-dir", str(config), "--analysis-path", str(args.analysis_path), "--limit", str(args.limit), "--max-random-fraction", str(args.max_random_fraction), "--max-poison-fraction", str(args.max_poison_fraction), "--seed", str(args.seed), "--bins", str(args.bins)]),
        ("audit", "audit_plans.py", ["--plans", str(plans), "--config-dir", str(config), "--analysis-path", str(args.analysis_path), "--output-dir", str(args.output_dir / "audit_permissible"), "--bins", str(args.bins)]),
        ("audit", "audit_plans.py", ["--plans", str(selected), "--config-dir", str(config), "--analysis-path", str(args.analysis_path), "--output-dir", str(args.output_dir / "audit_subsampled"), "--bins", str(args.bins)]),
        ("audit", "plot_fits.py", ["--output-dir", str(args.output_dir / "fits")]),
        ("audit", "plot_empirical_fits.py", ["--output-dir", str(args.output_dir / "empirical_fits")]),
    ]
    subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=root, check=True)
    start = next(index for index, (stage, _, _) in enumerate(jobs) if stage == args.start_at)
    environment = os.environ | {"MPLCONFIGDIR": "/tmp/rime-reproduction-matplotlib", "TQDM_DISABLE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    results = []
    for index, (stage, script, arguments) in enumerate(tqdm(jobs[start:], desc="Reproducing priors"), start=start):
        command = [sys.executable, str(root / "scripts" / script)] + arguments
        log_path = args.output_dir / ("%02d_%s.log" % (index, script.removesuffix(".py")))
        tqdm.write("Running %s; log: %s" % (script, log_path))
        started = time.monotonic()
        with log_path.open("w") as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        results.append({"stage": stage, "command": command, "seconds": time.monotonic() - started, "log": str(log_path)})
        (args.output_dir / "reproduction.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
