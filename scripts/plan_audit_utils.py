import json
import math
import numpy as np
import yaml
from pathlib import Path
from collections import Counter
from typing import Any
from scipy.special import betainc
from scipy.stats import truncnorm
from tqdm import tqdm


def load_configuration(
    config_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    distributions = yaml.safe_load((config_dir / "distributions.yaml").read_text())[
        "distributions"
    ]
    priors = flatten_distributions(distributions)
    recipes = yaml.safe_load((config_dir / "recipes.yaml").read_text())["recipes"]
    motifs = yaml.safe_load((config_dir / "motifs.yaml").read_text())["motifs"]
    routes = {}
    models = {}
    for recipe in tqdm(recipes, desc="Reading parameter definitions"):
        graphs = [("graph_spec", None, recipe["graph"])]
        graphs.extend(
            ("poison_graph_spec", poison["id"], poison["graph"])
            for poison in recipe.get("poisons", [])
        )
        for graph, poison_id, nodes in graphs:
            definitions = {}
            for node in nodes:
                expanded = (
                    dict(motifs[node["chain_ref"]]) | node
                    if "chain_ref" in node
                    else node
                )
                for location, spec in graph_parameters([expanded]):
                    if location[-1] == "sample":
                        for parameter, (name, model) in marginal_models(
                            spec, priors
                        ).items():
                            key = location[:-1] + (parameter,)
                            definitions[key] = {"sample": name}
                            models[name] = model
                            priors[name] = model
                    else:
                        definitions[location] = spec
            routes[(recipe["id"], graph, poison_id)] = definitions
        routes[(recipe["id"], "bindings", None)] = recipe.get("bindings", {})
    # Keep unobserved priors in the report, including discrete and authored priors.
    for reference in list(priors):
        for name, model in marginal_models(reference, priors).values():
            models[name] = model
    return routes, priors, models


def flatten_distributions(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    if "type" in tree or "values" in tree:
        return {prefix: tree}
    result = {}
    for key, value in tree.items():
        result.update(
            flatten_distributions(value, "%s.%s" % (prefix, key) if prefix else key)
        )
    return result


def marginal_models(
    reference: str,
    priors: dict[str, Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    model = priors[reference]
    if "sample" in model:
        return marginal_models(model["sample"], priors)
    if model["type"] == "parameters":
        result = {}
        for parameter, child in model["parameters"].items():
            name = "%s.%s" % (reference, parameter)
            if "sample" in child:
                result[parameter] = next(
                    iter(marginal_models(child["sample"], priors).values())
                )
            else:
                result[parameter] = (name, child)
        return result
    if model["type"] in {"mixture", "joint"}:
        components = []
        for part in model["components"]:
            child = (
                part["distribution"]
                if "distribution" in part
                else {"type": "parameters", "parameters": part["parameters"]}
            )
            local = dict(priors)
            local[reference] = child
            components.append((part["weight"], marginal_models(reference, local)))
        result = {}
        for parameter in components[0][1]:
            parts = [
                {"weight": weight, "model": children[parameter][1]}
                for weight, children in components
            ]
            name = reference if parameter == "" else "%s.%s" % (reference, parameter)
            result[parameter] = (
                name,
                {"type": "marginal_mixture", "components": parts},
            )
        return result
    return {"": (reference, model)}


def graph_parameters(
    nodes: list[dict[str, Any]],
    provenance: bool = False,
) -> list[tuple[tuple[str, str, str, str], Any]]:
    result = []
    params_key = "parameter_specs" if provenance else "params"
    for node in nodes:
        node_id = node.get("prefix", node.get("name", ""))
        for parameter, spec in node.get(params_key, {}).items():
            result.append(((node_id, "", node.get("operator", ""), parameter), spec))
        for parameter in ("send_level", "return_level", "dry_level"):
            if parameter in node and not provenance:
                result.append(((node_id, "", "", parameter), node[parameter]))
        for step in node.get("steps", []):
            for parameter, spec in step.get(params_key, {}).items():
                result.append(
                    ((node_id, step["name"], step["operator"], parameter), spec)
                )
    return result


def load_analysis(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open() as handle:
        return {
            row["clip_id"]: row["analysis"]
            for row in (
                json.loads(line)
                for line in tqdm(handle, desc="Reading analysis")
                if line.strip()
            )
        }


def context_value(
    path: str,
    plan: dict[str, Any],
    analysis: dict[str, Any] | None,
) -> Any:
    context = {"bindings": plan.get("bindings", {}), "metadata": {"analysis": analysis}}
    value = context
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def resolve_parameter(
    spec: Any,
    value: Any,
    name: str,
    bindings: dict[str, Any],
    priors: dict[str, Any],
    plan: dict[str, Any],
    analysis: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None, Any, str]:
    if isinstance(spec, dict) and "definition" in spec:
        name = spec["name"]
        spec = spec["definition"]
    if not isinstance(spec, dict):
        return name, {"type": "choice", "values": [spec]}, value, "fixed"
    if "sample" in spec:
        reference, model = next(iter(marginal_models(spec["sample"], priors).values()))
        return reference, model, value, "prior"
    if "values" in spec:
        choices = [
            item for item in spec["values"]
            if (not spec.get("include") or item in spec["include"])
            and item not in spec.get("exclude", [])
        ]
        return name, {"type": "choice", "values": choices}, value, "authored"
    if "ref" in spec:
        path = spec["ref"]
        if path.startswith("bindings.") and path.split(".", 1)[1] in bindings:
            return resolve_parameter(
                bindings[path.split(".", 1)[1]],
                value,
                name,
                bindings,
                priors,
                plan,
                analysis,
            )
        expected = context_value(path, plan, analysis)
        if expected is not None:
            return name, {"type": "choice", "values": [expected]}, value, "context"
        return name, None, value, "Missing context: %s" % path
    if "coalesce" in spec:
        for branch in spec["coalesce"]:
            if isinstance(branch, dict) and "ref" in branch:
                if branch["ref"].startswith("metadata.") and analysis is None:
                    return name, None, value, "Missing analysis for coalesce"
                if context_value(branch["ref"], plan, analysis) is None:
                    continue
            if branch is not None:
                return resolve_parameter(
                    branch, value, name, bindings, priors, plan, analysis
                )
    if "scale" in spec:
        scale = spec["scale"]
        factor = float(scale.get("factor", 1.0))
        if factor == 0:
            return (
                name,
                {"type": "choice", "values": [scale.get("offset", 0.0)]},
                value,
                "fixed",
            )
        original = (float(value) - float(scale.get("offset", 0.0))) / factor
        return resolve_parameter(
            scale["value"], original, name, bindings, priors, plan, analysis
        )
    if "tempo_sync" in spec:
        tempo = spec["tempo_sync"]
        bpm_spec = tempo["bpm"]
        if analysis is None and isinstance(bpm_spec, dict):
            return name, None, value, "Missing analysis for tempo_sync"
        branches = (
            bpm_spec.get("coalesce", [bpm_spec])
            if isinstance(bpm_spec, dict)
            else [bpm_spec]
        )
        bpm = next(
            context_value(branch["ref"], plan, analysis)
            if isinstance(branch, dict)
            else branch
            for branch in branches
            if not isinstance(branch, dict)
            or context_value(branch["ref"], plan, analysis) is not None
        )
        beats = float(value) * float(bpm) / 60.0
        return resolve_parameter(
            tempo["beats"], beats, name, bindings, priors, plan, analysis
        )
    return name, None, value, "No scalar prior for expression"


def extract_parameters(
    plan: dict[str, Any],
    routes: dict[str, Any],
    priors: dict[str, Any],
    analysis: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows, missing = [], []
    recipe = plan["recipe_id"]
    provenance = {
        graph: dict(graph_parameters(plan.get(graph) or [], provenance=True))
        for graph in ("graph_spec", "poison_graph_spec")
    }
    if (recipe, "graph_spec", None) not in routes and not any(provenance.values()):
        missing.append(
            {
                "graph": "graph_spec",
                "reason": "Recipe absent from supplied configuration",
            }
        )
    bindings = routes.get((recipe, "bindings", None), {}) | plan.get("binding_specs", {})
    for graph in ("graph_spec", "poison_graph_spec", "bindings"):
        poison = plan.get("poison_id") if graph == "poison_graph_spec" else None
        definitions = routes.get((recipe, graph, poison), {})
        if graph == "bindings":
            actual = [
                (("", "", "", key), value)
                for key, value in plan.get("bindings", {}).items()
                if not isinstance(value, (dict, list))
            ]
            definitions = {
                ("", "", "", key): value
                for key, value in bindings.items()
                if not isinstance(value, dict) or "each_from" not in value
            }
        else:
            actual = graph_parameters(plan.get(graph) or [])
            definitions = definitions | provenance[graph]
        seen = set()
        location_counts = Counter(location for location, _ in actual)
        for location, value in actual:
            seen.add(location)
            name = "settings.%s.%s.%s" % (
                recipe,
                graph if poison is None else "%s.%s" % (graph, poison),
                ".".join(location),
            )
            if location_counts[location] > 1:
                reference, model, transformed, status = (
                    name,
                    None,
                    value,
                    "Ambiguous graph parameter location",
                )
            elif location in definitions:
                reference, model, transformed, status = resolve_parameter(
                    definitions[location], value, name, bindings, priors, plan, analysis
                )
            else:
                reference, model, transformed, status = (
                    name,
                    None,
                    value,
                    "No configured parameter definition",
                )
            rows.append(
                {
                    "graph": graph,
                    "node": location[0],
                    "step": location[1],
                    "operator": location[2],
                    "parameter": location[3],
                    "prior": reference,
                    "value": transformed,
                    "resolved_value": value,
                    "model": model,
                    "status": status,
                }
            )
        if graph != "bindings" and (graph == "graph_spec" or plan.get(graph)):
            for location in definitions.keys() - seen:
                missing.append(
                    {
                        "graph": graph,
                        "node": location[0],
                        "step": location[1],
                        "parameter": location[3],
                        "reason": "Configured parameter missing from graph",
                    }
                )
    return rows, missing


def discrete_probabilities(model: dict[str, Any]) -> list[tuple[Any, float]] | None:
    kind = model["type"]
    if kind == "marginal_mixture":
        children = [
            (part["weight"], discrete_probabilities(part["model"]))
            for part in model["components"]
        ]
        if any(child is None for _, child in children):
            return None
        total = sum(weight for weight, _ in children)
        merged = {}
        for weight, child in children:
            for value, probability in child:
                key = json.dumps(value, sort_keys=True)
                previous = merged.get(key, (value, 0.0))[1]
                merged[key] = (value, previous + weight / total * probability)
        return list(merged.values())
    if kind == "int_uniform":
        values = list(range(int(model["low"]), int(model["high"]) + 1))
        return [(value, 1.0 / len(values)) for value in values]
    if kind not in {"choice", "grid", "values"}:
        return None
    parts = [
        (item["value"], float(item.get("weight", 1.0)))
        if isinstance(item, dict) and "value" in item
        else (item, 1.0)
        for item in model["values"]
    ]
    total = sum(weight for _, weight in parts)
    assert total > 0 and all(weight >= 0 for _, weight in parts)
    return [(value, weight / total) for value, weight in parts]


def category_index(value: Any, categories: list[tuple[Any, float]]) -> int:
    for index, (expected, _) in enumerate(categories):
        if isinstance(value, (int, float)) and isinstance(expected, (int, float)):
            equal = math.isclose(value, expected, rel_tol=1e-8, abs_tol=1e-8)
        else:
            equal = value == expected
        if equal:
            return index
    return -1


def support_bounds(model: dict[str, Any]) -> tuple[float, float]:
    if model["type"] == "marginal_mixture":
        bounds = [support_bounds(part["model"]) for part in model["components"]]
        return min(low for low, _ in bounds), max(high for _, high in bounds)
    return model["low"], model["high"]


def in_support(model: dict[str, Any], value: Any) -> bool:
    categories = discrete_probabilities(model)
    if categories is not None:
        index = category_index(value, categories)
        return index >= 0 and categories[index][1] > 0
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    if model["type"] == "marginal_mixture":
        return any(in_support(part["model"], value) for part in model["components"])
    low, high = support_bounds(model)
    return low <= value <= high


def evaluate_cdf(model: dict[str, Any], values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    kind = model["type"]
    if kind == "marginal_mixture":
        total = sum(part["weight"] for part in model["components"])
        return sum(
            part["weight"] / total * evaluate_cdf(part["model"], values)
            for part in model["components"]
        )
    if kind == "gaussian_mixture":
        total = sum(part["weight"] for part in model["components"])
        return sum(
            part["weight"]
            / total
            * evaluate_cdf(model | part | {"type": "normal"}, values)
            for part in model["components"]
        )
    low, high = support_bounds(model)
    logarithmic = model.get("scale") == "log" or kind in {"log_uniform", "power_law"}
    clipped = np.clip(values, low, high)
    x = np.log(clipped) if logarithmic else clipped
    a, b = (math.log(low), math.log(high)) if logarithmic else (low, high)
    if kind in {"uniform", "log_uniform"}:
        result = (x - a) / (b - a)
    elif kind == "normal":
        result = truncnorm.cdf(
            x,
            (a - model["mean"]) / model["std"],
            (b - model["mean"]) / model["std"],
            loc=model["mean"],
            scale=model["std"],
        )
    elif kind == "beta":
        result = betainc(model["alpha"], model["beta"], (x - a) / (b - a))
    elif kind == "power_law":
        exponent = model["exponent"] + 1
        result = (
            np.log(clipped / low) / math.log(high / low)
            if abs(exponent) < 1e-10
            else np.expm1(exponent * np.log(clipped / low))
            / math.expm1(exponent * math.log(high / low))
        )
    else:
        assert kind == "histogram", kind
        edges = np.asarray(model["edges"])
        edges = np.log(edges) if logarithmic else edges
        weights = np.asarray(model["weights"])
        result = np.interp(x, edges, np.r_[0.0, np.cumsum(weights) / weights.sum()])
    return np.where(values < low, 0.0, np.where(values > high, 1.0, result))


def coverage_bins(
    model: dict[str, Any],
    values: list[Any],
    bins: int,
) -> list[str | None]:
    categories = discrete_probabilities(model)
    if categories is not None:
        return [
            "category:%s" % json.dumps(categories[category_index(value, categories)][0], sort_keys=True)
            if in_support(model, value)
            else None
            for value in values
        ]
    valid = np.array([in_support(model, value) for value in values])
    result = np.full(len(values), -1, dtype=int)
    if valid.any():
        probabilities = evaluate_cdf(
            model,
            np.asarray(
                [value for value, keep in zip(values, valid) if keep], dtype=float
            ),
        )
        result[valid] = np.minimum((probabilities * bins).astype(int), bins - 1)
    return ["quantile:%d" % value if value >= 0 else None for value in result]
