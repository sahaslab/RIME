import sys
import argparse
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_truth.io_utils import write_json, load_records
from ground_truth.operators import load_operator_registry
from typing import Any


# Map block kinds to plan-level operator extraction behavior
def _operators_for_separate_block(block: Mapping[str, Any]) -> list[str]:
    return [block.get("operator", "separate_audio")]


def _operators_for_mix_block(block: Mapping[str, Any]) -> list[str]:
    return [block.get("operator", "mix_stems")]


def _operators_for_step(block: Mapping[str, Any]) -> list[str]:
    return [block["operator"]]


def _operators_for_chain_or_send_return_block(block: Mapping[str, Any]) -> list[str]:
    return [step["operator"] for step in block.get("steps", [])]


PLAN_OPERATOR_EXTRACTORS = {
    "separate": _operators_for_separate_block,
    "mix": _operators_for_mix_block,
    "step": _operators_for_step,
    "chain": _operators_for_chain_or_send_return_block,
    "send_return": _operators_for_chain_or_send_return_block
}


# Map block kinds to plan-level parameter extraction behavior
def _params_for_chain_block(block: Mapping[str, Any]) -> list[tuple[str, str, Any]]:
    return [
        (step["operator"], param_name, param_value)
        for step in block.get("steps", [])
        for param_name, param_value in step.get("params", {}).items()
    ]


def _params_for_step_block(block: Mapping[str, Any]) -> list[tuple[str, str, Any]]:
    return [
        (block["operator"], param_name, param_value)
        for param_name, param_value in block.get("params", {}).items()
    ]


PLAN_PARAM_EXTRACTORS = {
    "chain": _params_for_chain_block,
    "send_return": _params_for_chain_block,
    "step": _params_for_step_block
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans-path", type=Path, default=Path("derived/ground_truth/permissible_plans.jsonl"), help="Input JSONL of symbolic plans. Default: %(default)s")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/ground_truth"), help="Ground-truth config directory. Default: %(default)s")
    parser.add_argument("--output-path", type=Path, default=Path("derived/ground_truth/plan_coverage.json"), help="Output JSON report path. Default: %(default)s")
    args = parser.parse_args()

    rows = load_records(args.plans_path)
    operator_registry = load_operator_registry(args.config_dir)
    with (args.config_dir / "recipes.yaml").open("r", encoding="utf-8") as handle:
        recipe_config = yaml.safe_load(handle) or {}

    write_json(
        args.output_path,
        compute_coverage(
            rows=rows,
            recipe_config=recipe_config,
            operator_registry=operator_registry
        )
    )


def compute_coverage(
    rows: Sequence[Mapping[str, Any]],
    recipe_config: Mapping[str, Any],
    operator_registry: Any
) -> dict[str, Any]:
    clip_ids = {row["clip_id"] for row in rows}
    recipe_counter = Counter(row["recipe_id"] for row in rows)
    target_family_counter = Counter(
        row["target_family"] for row in rows if row.get("target_family") is not None
    )
    genre_counter = Counter()
    block_kind_counter = Counter()
    operator_counter = Counter()
    applied_policy_counter = Counter()
    parameter_counter: dict[str, Counter] = defaultdict(Counter)
    graph_sizes: list[int] = []
    plans_per_clip = Counter(row["clip_id"] for row in rows)

    for row in rows:
        genre_counter.update(row.get("genres", []))
        applied_policy_counter.update(row.get("applied_policies", []))
        operators = list(iter_plan_operators(row["graph_spec"]))
        operator_counter.update(operators)
        graph_sizes.append(len(operators))
        for block in row["graph_spec"]:
            block_kind_counter.update([block["kind"]])
        for operator_name, param_name, param_value in iter_plan_params(row["graph_spec"]):
            parameter_counter["%s.%s" % (operator_name, param_name)].update([str(param_value)])

    configured_recipe_ids = {recipe["id"] for recipe in recipe_config.get("recipes", [])}
    configured_operator_ids = set(operator_registry.names())

    return {
        "summary": {
            "num_clips": len(clip_ids),
            "num_plans": len(rows),
            "mean_plans_per_clip": safe_mean(list(plans_per_clip.values())),
            "mean_operators_per_plan": safe_mean(graph_sizes),
            "median_operators_per_plan": safe_median(graph_sizes)
        },
        "usage": {
            "recipes": sort_counter(recipe_counter),
            "operators": sort_counter(operator_counter),
            "target_families": sort_counter(target_family_counter),
            "genres": sort_counter(genre_counter),
            "block_kinds": sort_counter(block_kind_counter),
            "applied_policies": sort_counter(applied_policy_counter)
        },
        "plans_per_clip": sort_counter(plans_per_clip),
        "dead_recipes": sorted(configured_recipe_ids - set(recipe_counter.keys())),
        "dead_operators": sorted(configured_operator_ids - set(operator_counter.keys())),
        "parameter_usage": {
            key: sort_counter(counter)
            for key, counter in sorted(parameter_counter.items())
        }
    }


def iter_plan_operators(graph_spec: Sequence[Mapping[str, Any]]) -> list[str]:
    operators: list[str] = []
    for block in graph_spec:
        extractor = PLAN_OPERATOR_EXTRACTORS.get(block["kind"])
        if extractor is None:
            continue
        operators.extend(extractor(block))
    return operators


def iter_plan_params(graph_spec: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, Any]]:
    rows: list[tuple[str, str, Any]] = []
    for block in graph_spec:
        extractor = PLAN_PARAM_EXTRACTORS.get(block["kind"])
        if extractor is None:
            continue
        rows.extend(extractor(block))
    return rows


def safe_mean(values: Sequence[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def safe_median(values: Sequence[int]) -> float:
    return float(statistics.median(values)) if values else 0.0


def sort_counter(counter: Mapping[Any, int]) -> dict[str, int]:
    return {
        str(key): int(counter[key])
        for key in sorted(counter.keys(), key=lambda key: (-counter[key], str(key)))
    }


if __name__ == "__main__":
    main()
