"""Turn the sampling priors in distributions.yaml into UI-ready parameter controls.

Every control carries the support to clamp a slider to, the central value to open
it at, and the weighted options behind a dropdown, so a caller can offer exactly
the values the pipeline could have sampled and no others.

Effects are addressed the way recipes.yaml addresses them -- recipe, then step --
because distributions.yaml is organized by workflow family rather than by
operator. A parameter's prior is only discoverable through the step that samples
it, so there is no useful "list the effects" view that skips the recipe.

Deliberately torch-free: standard library and yaml only. graph/edit_graph.py
imports torch at module scope and ground_truth/runtime.py and planner.py pull it
in transitively, which is far too heavy for reading config.

Distribution semantics mirror GroundTruthPlanner's handlers
(ground_truth/planner.py:1557-1682). Central values are what
`_resolve_fitted_distribution` returns at probability 0.5 -- its middle
enumerate-mode quantile -- so a control's default is the same "typical" value
the planner would enumerate.
"""

import math
import argparse
from pathlib import Path
from statistics import NormalDist
from collections.abc import Mapping, Sequence
import yaml
from typing import Any

# Config sources this module reads, mirroring CONFIG_SOURCES in planner.py:21.
CONFIG_SOURCES = (
    ("distributions", "distributions.yaml", "distributions", {}),
    ("motifs", "motifs.yaml", "motifs", {}),
    ("recipes", "recipes.yaml", "recipes", [])
)

# Operators that are routing rather than effects; the graph emits them itself.
# Same set as prompt_lab_ui.py:55.
STRUCTURAL_OPERATORS = {"separate_audio", "mix_stems"}

# Block kinds that carry effect steps. `separate` and `mix` are routing, and a
# caller working on a single uploaded file has no stems to route.
EFFECT_BLOCK_KINDS = {"chain", "send_return"}

# Send/return bus trims, and their defaults in EditGraph.add_send_return
# (graph/edit_graph.py:265).
BUS_LEVEL_PARAMS = ("dry_level", "send_level", "return_level")

# Params a send bus pins fully wet, because add_send_return already passes the
# dry signal around the chain. Blending dry a second time doubles it. Same table
# as prompt_lab_ui.py:430.
BUS_PINNED_PARAMS = {"dry_level": 0.0, "wet_level": 1.0, "mix": 1.0}

# Distribution kinds resolved by a single fitted-quantile walk.
FITTED_KINDS = {"normal", "histogram", "log_uniform"}

# Distribution kinds whose `values` list makes them a dropdown.
CHOICE_KINDS = {"choice", "grid", "values"}

# Slider granularity. The priors are continuous, so this is only a UI step.
SLIDER_STEPS = 200

# The special param forms a recipe may use in place of a literal, from
# SPECIAL_VALUE_HANDLER_NAMES (planner.py:40).
SPECIAL_VALUE_KEYS = ("coalesce", "ref", "sample", "scale", "tempo_sync")

# Control id used for a joint distribution's mixture-component picker.
COMPONENT_CONTROL_ID = "__component__"


def load_yaml(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_configs(config_dir: Path | str) -> dict[str, Any]:
    """Read the config sources this module needs, keyed by bundle field name."""
    config_root = Path(config_dir)
    bundle: dict[str, Any] = {}
    for field_name, file_name, root_key, default_value in CONFIG_SOURCES:
        parsed = load_yaml(config_root / file_name)
        bundle[field_name] = parsed.get(root_key, default_value)
    return bundle


def distribution_leaves(distributions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten distributions.yaml to `dotted.path -> leaf spec`.

    Leaves are the nodes carrying `type`; everything above them is grouping, and
    only a leaf can be named by a `sample:` reference.
    """
    leaves: dict[str, dict[str, Any]] = {}

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, Mapping):
            return
        if "type" in node:
            leaves[prefix] = dict(node)
            return
        for key, child in node.items():
            walk(child, "%s.%s" % (prefix, key) if prefix else str(key))

    walk(distributions, "")
    return leaves


# ----------------------------------------------------------------------
# Distribution math. Mirrors planner.py:1557-1682.
# ----------------------------------------------------------------------
def choice_options(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Weighted options for a choice distribution, in config order.

    Bare scalars are accepted alongside the `{value, weight}` form, matching
    `_choice_values` (planner.py:1651).
    """
    options: list[dict[str, Any]] = []
    for index, item in enumerate(spec.get("values", [])):
        if isinstance(item, Mapping) and "value" in item:
            value, weight = item["value"], float(item.get("weight", 1.0))
        else:
            value, weight = item, 1.0
        options.append({"index": index, "value": value, "weight": weight})
    if not options:
        raise ValueError("Choice distribution requires at least one value.")
    return options


def fitted_quantile(spec: Mapping[str, Any], probability: float) -> float:
    """Inverse CDF of a normal, histogram or log_uniform leaf.

    A verbatim port of `_resolve_fitted_distribution` (planner.py:1599) for a
    single probability, so a control's bounds and default agree exactly with
    what the planner would sample or enumerate.
    """
    kind = str(spec["type"])
    logarithmic = spec.get("scale") == "log" or kind == "log_uniform"
    low, high = float(spec["low"]), float(spec["high"])
    if not low < high:
        raise ValueError("Distribution needs low < high, got low=%r high=%r." % (low, high))
    if logarithmic and low <= 0:
        raise ValueError("Log-scaled distribution needs low > 0, got %r." % low)
    lower_bound, upper_bound = (math.log(low), math.log(high)) if logarithmic else (low, high)

    if kind == "log_uniform":
        value = lower_bound + probability * (upper_bound - lower_bound)
    elif kind == "normal":
        normal = NormalDist(float(spec["mean"]), float(spec["std"]))
        lower, upper = normal.cdf(lower_bound), normal.cdf(upper_bound)
        if not lower < upper:
            raise ValueError("Normal distribution has empty truncation interval.")
        quantile = min(1.0 - 1e-15, max(1e-15, lower + probability * (upper - lower)))
        value = normal.inv_cdf(quantile)
    elif kind == "histogram":
        edges = [float(edge) for edge in spec["edges"]]
        edges = [math.log(edge) for edge in edges] if logarithmic else edges
        weights = [float(weight) for weight in spec["weights"]]
        if len(edges) != len(weights) + 1:
            raise ValueError("Histogram needs len(edges) == len(weights) + 1.")
        if not all(left < right for left, right in zip(edges, edges[1:])):
            raise ValueError("Histogram edges must be strictly increasing.")
        if not (all(weight >= 0 for weight in weights) and sum(weights) > 0):
            raise ValueError("Histogram weights must be non-negative with a positive sum.")
        remaining = probability * sum(weights)
        for index, weight in enumerate(weights):
            if weight > 0 and remaining <= weight:
                value = edges[index] + remaining / weight * (edges[index + 1] - edges[index])
                break
            remaining -= weight
        else:
            value = edges[-1]
    else:
        raise ValueError("Not a fitted distribution kind: '%s'." % kind)
    return min(high, max(low, math.exp(value) if logarithmic else value))


def leaf_bounds(spec: Mapping[str, Any]) -> tuple[float, float] | None:
    """Numeric support of a leaf, or None when it has no continuous range."""
    kind = str(spec.get("type", "choice"))
    if kind in CHOICE_KINDS:
        numeric = [option["value"] for option in choice_options(spec) if _is_number(option["value"])]
        return (float(min(numeric)), float(max(numeric))) if numeric else None
    if kind in {"uniform", "int_uniform"} or kind in FITTED_KINDS:
        return float(spec["low"]), float(spec["high"])
    return None


def central_value(spec: Mapping[str, Any]) -> Any:
    """The value a control should open at: the distribution's middle.

    Continuous kinds give their median, a choice gives its heaviest option, and
    a joint gives its heaviest component's per-parameter medians.
    """
    kind = str(spec.get("type", "choice"))
    if kind in CHOICE_KINDS:
        options = choice_options(spec)
        return max(options, key=lambda option: option["weight"])["value"]
    if kind == "uniform":
        return (float(spec["low"]) + float(spec["high"])) / 2.0
    if kind == "int_uniform":
        return int(round((float(spec["low"]) + float(spec["high"])) / 2.0))
    if kind in FITTED_KINDS:
        return fitted_quantile(spec, 0.5)
    if kind == "joint":
        component = _heaviest_component(spec)[1]
        return {name: central_value(model) for name, model in component["parameters"].items()}
    raise ValueError("Unsupported distribution type '%s'." % kind)


def _heaviest_component(spec: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]]:
    components = list(spec["components"])
    if not components:
        raise ValueError("Joint distribution requires at least one component.")
    index = max(range(len(components)), key=lambda i: float(components[i].get("weight", 1.0)))
    return index, components[index]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


# ----------------------------------------------------------------------
# Control descriptors
# ----------------------------------------------------------------------
# Unit labels inferred from a param's name suffix, longest first.
UNIT_SUFFIXES = (("_seconds", "s"), ("_hz", "Hz"), ("_db", "dB"), ("_ms", "ms"), ("bpm", "BPM"))


def _units(param: str) -> str | None:
    for suffix, label in UNIT_SUFFIXES:
        if param.endswith(suffix):
            return label
    return None


def _base_control(param_id: str, **extra: Any) -> dict[str, Any]:
    control = {
        "id": param_id,
        "param": param_id.split(".")[-1],
        "label": param_id,
        "units": _units(param_id.split(".")[-1]),
        "source": None,
        "dist_type": None,
        "origin": "literal",
        "derived_via": None,
        "note": None
    }
    control.update(extra)
    return control


def control_for_leaf(
    param_id: str,
    path: str,
    spec: Mapping[str, Any],
    *,
    origin: str = "sample"
) -> dict[str, Any]:
    """Build the control for one distribution leaf.

    A `choice` is discrete so it becomes a dropdown; everything with a
    continuous support becomes a slider clamped to that support.
    """
    kind = str(spec.get("type", "choice"))
    if kind in CHOICE_KINDS:
        options = choice_options(spec)
        default_index = max(range(len(options)), key=lambda index: options[index]["weight"])
        return _base_control(
            param_id,
            kind="dropdown",
            source=path,
            dist_type=kind,
            origin=origin,
            options=options,
            default_index=default_index,
            default=options[default_index]["value"]
        )

    bounds = leaf_bounds(spec)
    if bounds is None:
        return _unresolved(param_id, "distribution '%s' has no numeric support" % kind, source=path)
    low, high = bounds
    integer = kind == "int_uniform"
    return _base_control(
        param_id,
        kind="slider",
        source=path,
        dist_type=kind,
        origin=origin,
        min=low,
        max=high,
        step=1 if integer else (high - low) / SLIDER_STEPS,
        default=central_value(spec),
        integer=integer
    )


def _literal_control(param_id: str, value: float) -> dict[str, Any]:
    return _base_control(param_id, kind="number", default=value, origin="literal", note="literal in the recipe")


def _derived_control(param_id: str, value: Any, derived_via: str, formula: str) -> dict[str, Any]:
    return _base_control(
        param_id,
        kind="readonly",
        default=value,
        origin="derived",
        derived_via=derived_via,
        note="derived: %s" % formula
    )


def _unresolved(param_id: str, reason: str, **extra: Any) -> dict[str, Any]:
    return _base_control(
        param_id,
        kind="readonly",
        default=None,
        origin="unresolved",
        note="%s; left at the operator default" % reason,
        **extra
    )


def _fmt(value: float) -> str:
    return ("%g" % value) if _is_number(value) else str(value)


def _pick(values: Mapping[str, Any], control: Mapping[str, Any]) -> Any:
    """Resolve one control to a value, preferring the caller's over the default.

    A dropdown travels as an index into `options` because a choice value can be
    a list (vocals.harmony.intervals holds interval sets), which does not
    survive an HTML option value. Sliders clamp to the distribution's support --
    offering only samplable values is the whole point of deriving bounds here.
    """
    raw = values.get(control["id"])
    if control["kind"] == "dropdown":
        options = control["options"]
        index = control["default_index"] if raw is None else int(raw)
        if not 0 <= index < len(options):
            raise ValueError("Option index %d out of range for '%s'." % (index, control["id"]))
        return options[index]["value"]
    if raw is None or control["kind"] == "readonly":
        return control.get("default")
    number = float(raw)
    low, high = control.get("min"), control.get("max")
    if low is not None and high is not None:
        number = min(float(high), max(float(low), number))
    return int(round(number)) if control.get("integer") else number


# ----------------------------------------------------------------------
# Param resolution
# ----------------------------------------------------------------------
def _special_key(value: Mapping[str, Any]) -> str | None:
    """The special form a param mapping uses, or None for a plain mapping.

    Single-key rule from `_special_value_key` (planner.py:1461).
    """
    if len(value) != 1:
        return None
    key = next(iter(value.keys()))
    return key if key in SPECIAL_VALUE_KEYS else None


def resolve_param(
    param_id: str,
    spec: Any,
    leaves: Mapping[str, Any],
    values: Mapping[str, Any],
    controls: list[dict[str, Any]],
    bindings: Mapping[str, Any]
) -> Any:
    """Walk one param spec, appending its controls and returning its value.

    One walk serves both jobs: pass an empty `values` to get the UI schema
    sitting at every distribution's central value, or pass the caller's control
    values to compute what to hand the operator. Splitting this into a
    build-controls function and an apply-values function would let the control
    ids drift from the values meant to feed them.

    Returns None when a param cannot be resolved without clip metadata, which
    the caller should read as "omit it and let the DSP function's own default
    stand".
    """
    if isinstance(spec, Mapping):
        key = _special_key(spec)
        if key == "sample":
            path = str(spec["sample"])
            leaf = leaves.get(path)
            if leaf is None:
                controls.append(_unresolved(param_id, "unknown distribution path '%s'" % path))
                return None
            control = control_for_leaf(param_id, path, leaf)
            controls.append(control)
            return _pick(values, control)
        if key == "tempo_sync":
            inner = spec["tempo_sync"]
            bpm = resolve_param("%s.bpm" % param_id, inner.get("bpm"), leaves, values, controls, bindings)
            beats = resolve_param("%s.beats" % param_id, inner.get("beats"), leaves, values, controls, bindings)
            if bpm is None or beats is None:
                return None
            result = (60.0 / float(bpm)) * float(beats)
            controls.append(_derived_control(param_id, result, "tempo_sync", "(60 / bpm) * beats"))
            return result
        if key == "scale":
            inner = spec["scale"]
            base = resolve_param("%s.value" % param_id, inner.get("value"), leaves, values, controls, bindings)
            if base is None:
                return None
            factor, offset = float(inner.get("factor", 1.0)), float(inner.get("offset", 0.0))
            result = (float(base) * factor) + offset
            controls.append(
                _derived_control(param_id, result, "scale", "value * %s + %s" % (_fmt(factor), _fmt(offset)))
            )
            return result
        if key == "coalesce":
            # An uploaded file carries no clip metadata, so a `ref` into
            # metadata never resolves and the planner's own _can_resolve check
            # (planner.py:1759) falls through to the next candidate. That is how
            # the 120 BPM fallback in the synced-delay motifs is reached.
            first_control = len(controls)
            for candidate in spec["coalesce"]:
                if isinstance(candidate, Mapping) and _special_key(candidate) == "ref":
                    if not _binding_spec(str(candidate["ref"]), bindings):
                        continue
                resolved = resolve_param(param_id, candidate, leaves, values, controls, bindings)
                for control in controls[first_control:]:
                    if control["id"] == param_id and control["origin"] == "literal":
                        control["note"] = "fallback value; a real plan reads this from the clip"
                return resolved
            controls.append(_unresolved(param_id, "every coalesce candidate needs clip metadata"))
            return None
        if key == "ref":
            return _resolve_ref(param_id, str(spec["ref"]), leaves, values, controls, bindings)
        controls.append(_unresolved(param_id, "unsupported param form %s" % sorted(spec)))
        return None
    if _is_number(spec):
        control = _literal_control(param_id, spec)
        controls.append(control)
        return _pick(values, control)
    controls.append(_base_control(param_id, kind="readonly", default=spec, note="literal in the recipe"))
    return spec


def _binding_spec(path: str, bindings: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The binding a `ref` names, when it is one this module can resolve.

    Only a direct `bindings.<name>` hit counts. A deeper path such as
    `bindings.target_candidate.stem` reaches into a per-clip candidate record
    that an upload does not have.
    """
    segments = path.split(".")
    if len(segments) != 2 or segments[0] != "bindings":
        return None
    spec = bindings.get(segments[1])
    if not isinstance(spec, Mapping):
        return None
    if _special_key(spec) == "sample" or "values" in spec:
        return spec
    return None


def _resolve_ref(
    param_id: str,
    path: str,
    leaves: Mapping[str, Any],
    values: Mapping[str, Any],
    controls: list[dict[str, Any]],
    bindings: Mapping[str, Any]
) -> Any:
    """Resolve a `ref` against the recipe's own bindings where that is possible.

    Most bindings are `each_from: metadata...` or chase another binding that is,
    so they stay unresolved; a `values:` or `sample:` binding is self-contained
    and becomes a real control.
    """
    spec = _binding_spec(path, bindings)
    if spec is None:
        controls.append(_unresolved(param_id, "resolved per clip from '%s'" % path))
        return None
    if "values" in spec:
        options = [{"index": index, "value": value, "weight": 1.0} for index, value in enumerate(spec["values"])]
        control = _base_control(
            param_id,
            kind="dropdown",
            source=path,
            origin="ref",
            options=options,
            default_index=0,
            default=options[0]["value"],
            note="from the recipe binding '%s'" % path
        )
        controls.append(control)
        return _pick(values, control)
    return resolve_param(param_id, spec, leaves, values, controls, bindings)


def _resolve_joint(
    path: str,
    leaf: Mapping[str, Any],
    values: Mapping[str, Any],
    controls: list[dict[str, Any]]
) -> dict[str, Any]:
    """Controls and kwargs for a joint distribution used as a whole params map.

    A `joint` leaf is a mixture over *correlated* parameters, and recipes attach
    it by putting `sample:` at the params level rather than under a param name
    (motifs.yaml:85), so the drawn dict becomes the step's entire params map.
    The component picker comes first; the parameters below it belong to whichever
    component is selected, which is the only correlation the prior actually
    captures -- within a component the planner draws independently
    (planner.py:1593).
    """
    components = list(leaf["components"])
    default_index = _heaviest_component(leaf)[0]
    picker = _base_control(
        COMPONENT_CONTROL_ID,
        label="mixture component",
        kind="dropdown",
        source=path,
        dist_type="joint",
        origin="joint",
        options=[
            {"index": index, "value": index, "weight": float(component.get("weight", 1.0))}
            for index, component in enumerate(components)
        ],
        default_index=default_index,
        default=default_index,
        note="%d correlated settings groups; the parameters below come from the selected one" % len(components)
    )
    picker["param"] = "component"
    controls.append(picker)

    index = int(_pick(values, picker))
    kwargs: dict[str, Any] = {}
    for name, model in components[index]["parameters"].items():
        control = control_for_leaf(name, "%s.components[%d].%s" % (path, index, name), model, origin="joint")
        controls.append(control)
        kwargs[name] = _pick(values, control)
    return kwargs


def _annotate_bus_pinned(controls: list[dict[str, Any]], effect: Mapping[str, Any]) -> None:
    """Flag the blend params a send bus pins, so the UI can explain them.

    add_send_return already carries the dry signal around the chain, so an fx
    that also blends its own dry would double it. The motifs pin reverb to
    wet_level 1.0 / dry_level 0.0 and delay to mix 1.0 for exactly that reason.
    """
    if effect.get("block_kind") != "send_return":
        return
    for control in controls:
        expected = BUS_PINNED_PARAMS.get(control["param"])
        if expected is None:
            continue
        control["bus_pinned"] = True
        control["note"] = "pinned to %s by the send bus, which supplies the dry path" % _fmt(expected)


def effect_params(
    effect: Mapping[str, Any],
    leaves: Mapping[str, Any],
    values: Mapping[str, Any],
    bindings: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Resolve an effect step's params to (kwargs, controls, notes)."""
    controls: list[dict[str, Any]] = []
    notes: list[str] = []
    spec = effect.get("params") or {}

    if isinstance(spec, Mapping) and _special_key(spec) == "sample":
        path = str(spec["sample"])
        leaf = leaves.get(path)
        if leaf is None:
            notes.append("unknown distribution path '%s'; every param left at its default" % path)
            return {}, controls, notes
        if str(leaf.get("type")) != "joint":
            notes.append("'%s' is a %s, but it is attached as a whole params map" % (path, leaf.get("type")))
            return {}, controls, notes
        return _resolve_joint(path, leaf, values, controls), controls, notes

    kwargs: dict[str, Any] = {}
    for param, value in spec.items():
        resolved = resolve_param(str(param), value, leaves, values, controls, bindings)
        if resolved is None:
            notes.append("%s left at the operator default" % param)
            continue
        kwargs[str(param)] = resolved
    return kwargs, controls, notes


def bus_levels(
    effect: Mapping[str, Any],
    leaves: Mapping[str, Any],
    values: Mapping[str, Any],
    bindings: Mapping[str, Any]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Resolve a send_return block's trims to (levels, controls).

    Defaults match add_send_return's own (graph/edit_graph.py:265): a trim the
    block omits is unity.
    """
    bus = effect.get("bus")
    if not bus:
        return {}, []
    controls: list[dict[str, Any]] = []
    levels: dict[str, float] = {}
    for name in BUS_LEVEL_PARAMS:
        resolved = resolve_param("bus.%s" % name, bus.get(name, 1.0), leaves, values, controls, bindings)
        levels[name] = 1.0 if resolved is None else float(resolved)
    for control in controls:
        control["origin"] = "bus"
    return levels, controls


def effect_schema(
    effect: Mapping[str, Any],
    leaves: Mapping[str, Any],
    bindings: Mapping[str, Any],
    values: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Everything a caller needs to draw, or to apply, one effect step."""
    values = values or {}
    levels, bus_controls = bus_levels(effect, leaves, values, bindings)
    params, controls, notes = effect_params(effect, leaves, values, bindings)
    _annotate_bus_pinned(controls, effect)
    return {
        "key": effect["key"],
        "block": effect["block"],
        "block_kind": effect["block_kind"],
        "step": effect["step"],
        "operator": effect["operator"],
        "motif": effect["motif"],
        "group": effect["group"],
        "bus": levels or None,
        "bus_controls": bus_controls,
        "controls": controls,
        "params": params,
        "notes": notes
    }


# ----------------------------------------------------------------------
# Walking recipes for their effect steps
# ----------------------------------------------------------------------
def _block_steps(block: Mapping[str, Any], motifs: Mapping[str, Any]) -> tuple[list[Any], str | None]:
    """A block's steps, following `chain_ref` into motifs.yaml.

    Mirrors `_normalize_block` (planner.py:1427).
    """
    if block.get("steps"):
        return list(block["steps"]), None
    reference = block.get("chain_ref")
    if not reference:
        return [], None
    motif = motifs.get(str(reference)) or {}
    return list(motif.get("steps", [])), str(reference)


def _graph_effects(
    graph: Sequence[Mapping[str, Any]],
    motifs: Mapping[str, Any],
    group: str,
    taken: set[str]
) -> list[dict[str, Any]]:
    """Every selectable effect step in one graph, in graph order."""
    effects: list[dict[str, Any]] = []
    for block in graph or []:
        kind = str(block.get("kind", ""))
        if kind not in EFFECT_BLOCK_KINDS:
            continue
        steps, motif = _block_steps(block, motifs)
        block_name = str(block.get("name") or block.get("prefix") or kind)
        bus = {name: block.get(name, 1.0) for name in BUS_LEVEL_PARAMS} if kind == "send_return" else None
        for step in steps:
            operator_name = str(step.get("operator", ""))
            if not operator_name or operator_name in STRUCTURAL_OPERATORS:
                continue
            step_name = str(step.get("name") or operator_name)
            key = "%s/%s" % (block_name, step_name)
            if key in taken:
                key = "%s#%d" % (key, len(taken))
            taken.add(key)
            effects.append(
                {
                    "key": key,
                    "block": block_name,
                    "block_kind": kind,
                    "step": step_name,
                    "operator": operator_name,
                    "motif": motif,
                    "group": group,
                    "bus": bus,
                    "params": dict(step.get("params", {}))
                }
            )
    return effects


def target_hint(recipe: Mapping[str, Any]) -> str | None:
    """Readable stem constraint for a recipe, from its `when` clause.

    An upload carries no clip metadata, so the stem a real plan would pick
    (`bindings.target_candidate` <- metadata.analysis.target_candidates) cannot
    be resolved here. The `when` conditions on bindings.target_family are the
    next best thing, since they are what constrains the stem recipe-wide.
    """
    requires: list[str] = []
    excludes: list[str] = []

    def walk(node: Any, negated: bool) -> None:
        if isinstance(node, Mapping):
            for key, child in node.items():
                if key in {"eq", "neq"} and isinstance(child, Mapping):
                    if str(child.get("path")) == "bindings.target_family":
                        positive = (key == "eq") != negated
                        (requires if positive else excludes).append(str(child.get("value")))
                        continue
                walk(child, not negated if key == "not" else negated)
        elif isinstance(node, list):
            for child in node:
                walk(child, negated)

    walk(recipe.get("when"), False)
    if requires:
        return " or ".join(sorted(set(requires)))
    if excludes:
        return "any stem except %s" % ", ".join(sorted(set(excludes)))
    return None


def resolve_recipe_effects(
    recipes: Sequence[Mapping[str, Any]],
    motifs: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Each recipe with the effect steps a caller can select, in config order."""
    resolved: list[dict[str, Any]] = []
    for recipe in recipes:
        taken: set[str] = set()
        effects = _graph_effects(recipe.get("graph", []), motifs, "edit", taken)
        for poison in recipe.get("poisons", []) or []:
            effects.extend(_graph_effects(poison.get("graph", []), motifs, "poison", taken))
        if not effects:
            continue
        resolved.append(
            {
                "id": str(recipe.get("id", "")),
                "description": str(recipe.get("description", "")),
                "tags": list(recipe.get("tags", [])),
                "target_hint": target_hint(recipe),
                "bindings": dict(recipe.get("bindings", {})),
                "effects": effects
            }
        )
    return resolved


# ----------------------------------------------------------------------
# Smoke check
# ----------------------------------------------------------------------
def _check_control(control: Mapping[str, Any], where: str) -> list[str]:
    """Problems with one control, as messages. Empty means it is usable."""
    problems: list[str] = []
    kind = control["kind"]
    if kind == "slider":
        low, high, default = control["min"], control["max"], control["default"]
        if not low < high:
            problems.append("%s: slider needs min < max, got %r..%r" % (where, low, high))
        if not _is_number(default):
            problems.append("%s: slider default is not a number: %r" % (where, default))
        elif not low <= default <= high:
            problems.append("%s: default %r outside support %r..%r" % (where, default, low, high))
        if not control["step"] > 0:
            problems.append("%s: slider step must be positive, got %r" % (where, control["step"]))
    elif kind == "dropdown":
        options = control["options"]
        if not options:
            problems.append("%s: dropdown has no options" % where)
        elif not 0 <= control["default_index"] < len(options):
            problems.append("%s: default_index %r out of range" % (where, control["default_index"]))
    elif kind == "number":
        if not _is_number(control["default"]):
            problems.append("%s: number default is not a number: %r" % (where, control["default"]))
    elif kind != "readonly":
        problems.append("%s: unknown control kind '%s'" % (where, kind))
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that every recipe effect yields a usable control set.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs" / "ground_truth",
        help="Ground-truth config directory. Default: %(default)s"
    )
    parser.add_argument("--verbose", action="store_true", help="Print every effect and control.")
    args = parser.parse_args()

    bundle = load_configs(args.config_dir)
    leaves = distribution_leaves(bundle["distributions"])
    recipes = resolve_recipe_effects(bundle["recipes"], bundle["motifs"])

    problems: list[str] = []
    effect_count = control_count = 0
    unresolved: list[str] = []
    kinds: dict[str, int] = {}

    for recipe in recipes:
        for effect in recipe["effects"]:
            effect_count += 1
            where = "%s/%s" % (recipe["id"], effect["key"])
            try:
                schema = effect_schema(effect, leaves, recipe["bindings"])
            except Exception as error:
                problems.append("%s: %s: %s" % (where, type(error).__name__, error))
                continue

            every_control = schema["bus_controls"] + schema["controls"]
            control_count += len(every_control)
            for control in every_control:
                kinds[control["kind"]] = kinds.get(control["kind"], 0) + 1
                problems.extend(_check_control(control, "%s %s" % (where, control["id"])))
                if control["origin"] == "unresolved":
                    unresolved.append("%s %s" % (where, control["id"]))

            # Feeding the schema's own defaults back in must reproduce it: the
            # build walk and the apply walk are the same code, so a mismatch
            # means a control id does not address the value it renders.
            defaults = {
                control["id"]: control.get("default_index") if control["kind"] == "dropdown" else control.get("default")
                for control in every_control
                if control["kind"] != "readonly"
            }
            replay = effect_schema(effect, leaves, recipe["bindings"], defaults)
            if replay["params"] != schema["params"]:
                problems.append("%s: replaying defaults changed params\n  %r\n  %r" % (where, schema["params"], replay["params"]))
            if replay["bus"] != schema["bus"]:
                problems.append("%s: replaying defaults changed bus %r -> %r" % (where, schema["bus"], replay["bus"]))
            if not schema["params"]:
                problems.append("%s: resolved to no params at all" % where)

            if args.verbose:
                print("%-58s %s" % (where, effect["operator"]))
                for control in every_control:
                    print("    %-26s %-9s %s" % (control["id"], control["kind"], _describe(control)))

    print()
    print("recipes            %d" % len(recipes))
    print("effect steps       %d" % effect_count)
    print("distribution leaves %d" % len(leaves))
    print("controls           %d  (%s)" % (control_count, ", ".join("%s %d" % item for item in sorted(kinds.items()))))
    if unresolved:
        print("needs clip metadata, left at operator defaults: %d" % len(unresolved))
        for entry in sorted(set(unresolved)):
            print("    %s" % entry)
    if problems:
        print()
        print("FAILED: %d problem(s)" % len(problems))
        for problem in problems:
            print("  %s" % problem)
        raise SystemExit(1)
    print()
    print("OK: every effect builds a usable control set and replays its own defaults")


def _describe(control: Mapping[str, Any]) -> str:
    if control["kind"] == "slider":
        return "%s..%s default %s%s" % (
            _fmt(control["min"]),
            _fmt(control["max"]),
            _fmt(control["default"]),
            " %s" % control["units"] if control["units"] else ""
        )
    if control["kind"] == "dropdown":
        return "%d options, default %r" % (len(control["options"]), control["default"])
    return "%s%s" % (_fmt(control.get("default")), " -- %s" % control["note"] if control.get("note") else "")


if __name__ == "__main__":
    main()
