import sys
import json
import random
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Any
from tqdm import tqdm
from plan_audit_utils import load_configuration, discrete_probabilities, category_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--analysis-path", type=Path, default=Path("derived/ground_truth/musiccaps_analysis_manifest.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("derived/ground_truth/sampling_checks.json"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.root))
    from extract_priors import model_cdf
    from ground_truth.planner import GroundTruthPlanner, DISTRIBUTION_HANDLER_NAMES
    priors = json.loads((args.root / "generation_priors/02_priors/priors.json").read_text())
    planner = GroundTruthPlanner.from_directory(args.root / "generation_priors/03_rime")
    checks = []
    models = [item for name, model in priors["distributions"].items() for item in scalar_models(name, model)]
    for name, model in tqdm(models, desc="Checking sampled CDFs"):
        rng = random.Random(123)
        handler = getattr(planner, DISTRIBUTION_HANDLER_NAMES[model["type"]])
        draws = np.sort([handler(model, "sample", rng)[0] for _ in range(args.draws)])
        assert np.isfinite(draws).all()
        assert draws.min() >= model["low"] - 1e-10 and draws.max() <= model["high"] + 1e-10
        cdf = model_cdf(model, draws)
        error = float(max(np.max(np.abs(cdf - np.arange(1, args.draws + 1) / args.draws)), np.max(np.abs(cdf - np.arange(args.draws) / args.draws))))
        assert error < 0.025, (name, error)
        assert handler(model, "enumerate", random.Random(123))
        checks.append({"prior": name, "family": model["type"], "draws": args.draws, "cdf_error": error})
    _, _, configured = load_configuration(args.root / "generation_priors/03_rime")
    for name, model in tqdm(configured.items(), desc="Checking discrete priors"):
        categories = discrete_probabilities(model)
        if categories is None:
            continue
        rng = random.Random(123)
        draws = [planner._distribution_values(name, "sample", rng)[0] for _ in range(args.draws)]
        counts = Counter(category_index(value, categories) for value in draws)
        assert counts[-1] == 0, name
        error = sum(abs(counts[index] / args.draws - probability) for index, (_, probability) in enumerate(categories)) / 2
        assert error < 0.025, (name, error)
        checks.append({"prior": name, "family": model["type"], "draws": args.draws, "total_variation": error})
    rng = random.Random(123)
    for name in ["vocals.compression.settings", "dynamics.compression.settings", "tone.equalization.highshelf.settings"]:
        for _ in tqdm(range(1000), desc="Checking %s" % name):
            values = planner._distribution_values(name, "sample", rng)[0]
            assert all(np.isfinite(value) for value in values.values())
            if "compression" in name:
                assert -40 <= values["threshold_db"] <= -10 and 1 <= values["ratio"] <= 10
                assert 5 <= values["attack_ms"] <= 10 and 40 <= values["release_ms"] <= 80
            else:
                assert 5 <= abs(values["gain_db"]) <= 18
    metadata = [json.loads(line) for line in args.analysis_path.read_text().splitlines()[:5]]
    plan_count = 0
    for row in tqdm(metadata, desc="Checking complete RIME plans"):
        plans = planner.plan(row, mode="sample", samples_per_recipe=2, seed=123)
        for plan in plans:
            planner.describe_plan(plan, validate=True)
        plan_count += len(plans)
    output = {"scalar_checks": checks, "validated_plans": plan_count, "checked_parameter_sets": 3000}
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2) + "\n")
    print("Validated %d scalar models, 3000 parameter sets, and %d complete plans." % (len(checks), plan_count))


def scalar_models(name: str, model: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "sample" in model:
        return []
    if model["type"] == "parameters":
        return [item for key, value in model["parameters"].items() for item in scalar_models(name + "." + key, value)]
    if model["type"] == "mixture":
        return [item for part in model["components"] for item in scalar_models(name + "." + part["label"], part["distribution"])]
    assert model["type"] in {"normal", "beta", "power_law", "gaussian_mixture", "uniform", "log_uniform"}
    return [(name, model)]


if __name__ == "__main__":
    main()
