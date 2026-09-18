import sys
import json
import random
import argparse
import pytest
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ground_truth.planner import GroundTruthPlanner
from plan_audit_utils import load_configuration, extract_parameters, in_support
from subsample_ground_truth_plans import read_candidates


@pytest.mark.parametrize("family", ["vocals", "drums", "bass", "guitar"])
@pytest.mark.parametrize("bpm", [None, 120.0])
def test_all_random_parameters_have_provenance(family: str, bpm: float | None) -> None:
    config = Path(__file__).resolve().parents[1] / "configs/ground_truth"
    planner = GroundTruthPlanner.from_directory(config)
    routes, priors, _ = load_configuration(config)
    metadata = {
        "analysis": {
            "tempo_bpm": bpm,
            "target_candidates": [{"stem": family, "family": family}],
        }
    }
    plans = planner.random_plans(metadata, count=12, seed=22)
    assert 0 < len(plans) <= 12
    for plan in tqdm(plans, desc="Checking random provenance"):
        row = json.loads(json.dumps(plan.to_dict()))
        values, missing = extract_parameters(row, routes, priors, None)
        assert not missing
        assert values
        assert all(item["model"] is not None for item in values)
        assert all(in_support(item["model"], item["value"]) for item in values)
        planner.describe_plan(plan, validate=True)


def test_delay_branches_gain_branches_and_category_credit(tmp_path: Path) -> None:
    config = Path(__file__).resolve().parents[1] / "configs/ground_truth"
    planner = GroundTruthPlanner.from_directory(config)
    routes, priors, _ = load_configuration(config)
    observed = set()
    records = []
    for seed in tqdm(range(20), desc="Checking prior branches"):
        for operator in ("apply_delay_effect", "apply_gain", "apply_peak_filter"):
            step = planner._random_step_for_operator(
                operator,
                {"analysis": {"tempo_bpm": 120.0}},
                {"family": "vocals"},
                random.Random(seed),
                1,
            )
            record = {
                "clip_id": str(seed),
                "recipe_id": "random_constrained",
                "graph_spec": [{"kind": "chain", "prefix": "test", "steps": [step]}],
            }
            rows, missing = extract_parameters(record, routes, priors, None)
            assert not missing
            observed.update(
                item["prior"]
                for item in rows
                if item["status"] in {"prior", "authored"}
            )
            for item in rows:
                assert in_support(item["model"], item["value"])
                if item["prior"] == "space.shared_send.delay.synced.beats":
                    assert item["value"] == item["resolved_value"] * 2
            records.append(record)
    assert {
        "space.shared_send.delay.synced.beats",
        "space.shared_send.delay.free.seconds",
        "balance.target_gain.up_db",
        "balance.target_gain.down_db",
        "random.apply_peak_filter.cutoff_frequency_hz",
    } <= observed
    path = tmp_path / "plans.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    args = argparse.Namespace(plans_path=path, input_limit=None, bins=10)
    _, features, unresolved = read_candidates(args, routes, priors, {})
    assert not unresolved
    assert observed <= {dimension for items in features for dimension, _ in items}


def test_tampered_random_parameter_is_outside_support() -> None:
    config = Path(__file__).resolve().parents[1] / "configs/ground_truth"
    planner = GroundTruthPlanner.from_directory(config)
    routes, priors, _ = load_configuration(config)
    step = planner._random_step_for_operator(
        "apply_limiter_effect", {}, {}, random.Random(0), 1
    )
    step["params"]["threshold_db"] = 1000.0
    record = {
        "recipe_id": "random_constrained",
        "graph_spec": [{"kind": "chain", "prefix": "test", "steps": [step]}],
    }
    rows, missing = extract_parameters(record, routes, priors, None)
    assert not missing
    threshold = next(item for item in rows if item["parameter"] == "threshold_db")
    assert not in_support(threshold["model"], threshold["value"])
    del step["params"]["release_ms"]
    assert extract_parameters(record, routes, priors, None)[1]
    del step["parameter_specs"]
    rows, missing = extract_parameters(record, routes, priors, None)
    assert missing
    assert all(item["model"] is None for item in rows)
