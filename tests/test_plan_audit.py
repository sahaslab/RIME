import sys
import json
import argparse
import random
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_plans import summarize
from plan_audit_utils import (
    load_configuration,
    extract_parameters,
    resolve_parameter,
    coverage_bins,
    evaluate_cdf,
)
from subsample_ground_truth_plans import select_candidates


def test_empty_target_candidates_do_not_crash_sampling() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ground_truth.planner import GroundTruthPlanner

    planner = GroundTruthPlanner.from_directory(Path(__file__).resolve().parents[1] / "configs/ground_truth")
    assert planner._binding_each_from(
        {"each_from": "metadata.analysis.target_candidates"},
        {"metadata": {"analysis": {"target_candidates": []}}},
        "sample",
        random.Random(0),
        "each_from",
    ) == []


def test_discrete_weights_and_invalid_categories() -> None:
    model = {
        "type": "choice",
        "values": [{"value": "soft", "weight": 3}, {"value": "hard", "weight": 1}],
    }
    values = ["soft", "soft", "soft", "hard"]
    rows = [
        {"value": value, "clip_id": str(index), "in_support": True}
        for index, value in enumerate(values)
    ]
    summary = summarize(rows, model, 10)
    assert summary["coverage"] == 1.0
    assert summary["total_variation"] == 0.0
    assert coverage_bins(model, ["soft", "hard", "unknown"], 10) == [
        "category:\"soft\"",
        "category:\"hard\"",
        None,
    ]


def test_continuous_mixture_and_quantile_bins() -> None:
    model = {
        "type": "marginal_mixture",
        "components": [
            {"weight": 3.0, "model": {"type": "uniform", "low": 0.0, "high": 1.0}},
            {"weight": 1.0, "model": {"type": "uniform", "low": 2.0, "high": 3.0}},
        ],
    }
    np.testing.assert_allclose(
        evaluate_cdf(model, np.array([0.5, 1.5, 2.5])), [0.375, 0.75, 0.875]
    )
    assert coverage_bins(model, [0.5, 1.5, 2.5, "invalid"], 4) == [
        "quantile:1",
        None,
        "quantile:3",
        None,
    ]


def test_tempo_and_metadata_override() -> None:
    priors = {"beats": {"type": "choice", "values": [0.5, 1.0]}}
    spec = {
        "tempo_sync": {
            "bpm": {"coalesce": [{"ref": "metadata.analysis.tempo_bpm"}, 120.0]},
            "beats": {"sample": "beats"},
        }
    }
    reference, model, value, status = resolve_parameter(
        spec, 0.5, "delay", {}, priors, {}, {"tempo_bpm": 60}
    )
    assert (reference, value, status) == ("beats", 0.5, "prior")
    assert model == priors["beats"]
    assert resolve_parameter(spec, 0.5, "delay", {}, priors, {}, None)[1] is None
    coalesce = {
        "coalesce": [{"ref": "metadata.analysis.interval"}, {"sample": "beats"}]
    }
    assert (
        resolve_parameter(coalesce, 7, "interval", {}, priors, {}, {"interval": 7})[3]
        == "context"
    )
    assert resolve_parameter(coalesce, 1, "interval", {}, priors, {}, {})[0] == "beats"


def test_graph_block_joint_binding_and_poison_parameters(tmp_path: Path) -> None:
    distributions = {
        "gain": {"type": "uniform", "low": 1.0, "high": 5.0},
        "settings": {
            "type": "parameters",
            "parameters": {"gain_db": {"sample": "gain"}},
        },
    }
    recipe = {
        "id": "test",
        "bindings": {"correction": {"sample": "gain"}},
        "graph": [
            {
                "kind": "send_return",
                "name": "bus",
                "send_level": {"sample": "gain"},
                "chain_ref": "gain_chain",
            }
        ],
        "poisons": [
            {
                "id": "quiet",
                "graph": [
                    {
                        "kind": "step",
                        "name": "poison",
                        "operator": "apply_gain",
                        "params": {
                            "gain_db": {
                                "scale": {
                                    "factor": -1.0,
                                    "value": {"ref": "bindings.correction"},
                                }
                            }
                        },
                    }
                ],
            }
        ],
    }
    (tmp_path / "distributions.yaml").write_text(
        json.dumps({"distributions": distributions})
    )
    (tmp_path / "recipes.yaml").write_text(json.dumps({"recipes": [recipe]}))
    (tmp_path / "motifs.yaml").write_text(
        json.dumps(
            {
                "motifs": {
                    "gain_chain": {
                        "steps": [
                            {
                                "name": "gain",
                                "operator": "apply_gain",
                                "params": {"sample": "settings"},
                            }
                        ]
                    }
                }
            }
        )
    )
    routes, priors, _ = load_configuration(tmp_path)
    plan = {
        "recipe_id": "test",
        "bindings": {"correction": 2.0},
        "graph_spec": [
            {
                "kind": "send_return",
                "name": "bus",
                "send_level": 3.0,
                "steps": [
                    {
                        "name": "gain",
                        "operator": "apply_gain",
                        "params": {"gain_db": 4.0},
                    }
                ],
            }
        ],
        "poison_id": "quiet",
        "poison_graph_spec": [
            {
                "kind": "step",
                "name": "poison",
                "operator": "apply_gain",
                "params": {"gain_db": -2.0},
            }
        ],
    }
    rows, missing = extract_parameters(plan, routes, priors, {})
    assert not missing
    assert len(rows) == 4
    assert {row["prior"] for row in rows} == {"gain"}
    poison = next(row for row in rows if row["graph"] == "poison_graph_spec")
    assert poison["value"] == 2.0 and poison["resolved_value"] == -2.0
    del plan["graph_spec"][0]["steps"][0]["params"]["gain_db"]
    assert len(extract_parameters(plan, routes, priors, {})[1]) == 1


def test_selector_fills_budget_without_discarding_bins() -> None:
    candidates = [
        {"clip_id": str(index // 2), "recipe_id": "recipe"} for index in range(8)
    ]
    features = [
        {("recipe", "recipe"), ("target_family", "vocals"), ("prior", str(index % 4))}
        for index in range(8)
    ]
    args = argparse.Namespace(
        seed=7, limit=4, max_per_source_recipe=1, policy="prior_coverage"
    )
    selected, _ = select_candidates(candidates, features, args)
    assert len(selected) == 4
    assert len({candidates[index]["clip_id"] for index in selected}) == 4
    assert {
        bucket
        for index in selected
        for dimension, bucket in features[index]
        if dimension == "prior"
    } == {"0", "1", "2", "3"}
    assert select_candidates(candidates, features, args)[0] == selected
