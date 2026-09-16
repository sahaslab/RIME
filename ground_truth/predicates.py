"""The condition DSL shared by plan generation and rejection filtering.

This is the `when:` language of `configs/ground_truth/constraints.yaml` and
`recipes.yaml`, lifted out of `GroundTruthPlanner` unchanged so a second stage
can evaluate rules against its own context without instantiating a planner.
Nothing here reads planner state; the methods it came from never did.

A condition is a single-key mapping. The key selects either a comparison
(`{"gte": {"path": ..., "value": ...}}`) or a handler
(`{"all": [...]}`, `{"contains_any": {"path": ..., "values": [...]}}`).

Two properties of this evaluator are load-bearing for rule authors, and both are
deliberate rather than accidental:

- A missing path is `False`, never an error. That is what lets a rule say "this
  clip has no tempo" without guarding every lookup, and it is why
  `not: {exists: ...}` is the way to spell "absent". The cost is that a typo'd
  path yields a rule that silently never fires, so a caller holding
  user-authored rules should validate their paths up front rather than trust
  them -- see `iter_condition_paths`.
- Comparisons are tested before handlers, and the first matching key wins. A
  mapping carrying two operator keys therefore evaluates only one of them, so
  conditions are written single-key.
"""

import operator
from collections.abc import Mapping, Sequence
from typing import Any

# Comparison operators available in condition specs, each taking
# `{"path": ..., "value": ...}`.
COMPARISON_OPERATORS = {
    "eq": operator.eq,
    "neq": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le
}


def lookup_path(context: Mapping[str, Any], path: str) -> Any:
    """The value at a dotted path, traversing mappings and object attributes."""
    current: Any = context
    for segment in path.split("."):
        if isinstance(current, Mapping):
            current = current[segment]
        else:
            current = getattr(current, segment)
    return current


def path_exists(context: Mapping[str, Any], path: str) -> bool:
    """Whether a dotted path resolves, without raising if it does not."""
    current: Any = context
    for segment in path.split("."):
        if isinstance(current, Mapping):
            if segment not in current:
                return False
            current = current[segment]
            continue
        if not hasattr(current, segment):
            return False
        current = getattr(current, segment)
    return True


def as_list(value: Any) -> list[Any]:
    """A value as a list, so a scalar and a list membership-test alike."""
    return value if isinstance(value, list) else [value]


def evaluate_condition(
    condition: Mapping[str, Any] | None,
    context: Mapping[str, Any]
) -> bool:
    """Whether a condition holds against a context. `None` is vacuously true."""
    if condition is None:
        return True
    for key, compare in COMPARISON_OPERATORS.items():
        if key in condition:
            return _condition_compare(condition[key], context, compare)
    for key, handler in CONDITION_HANDLERS.items():
        if key in condition:
            return handler(condition[key], context)
    raise ValueError("Unsupported condition '%s'." % condition)


def _condition_all(conditions: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> bool:
    return all(evaluate_condition(item, context) for item in conditions)


def _condition_any(conditions: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> bool:
    return any(evaluate_condition(item, context) for item in conditions)


def _condition_not(condition: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    return not evaluate_condition(condition, context)


def _condition_exists(spec: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    return path_exists(context, spec["path"])


def _condition_in(spec: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    if not path_exists(context, spec["path"]):
        return False
    return lookup_path(context, spec["path"]) in list(spec["values"])


def _condition_contains_any(spec: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    if not path_exists(context, spec["path"]):
        return False
    values = as_list(lookup_path(context, spec["path"]))
    return any(value in values for value in spec["values"])


def _condition_contains_all(spec: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
    if not path_exists(context, spec["path"]):
        return False
    values = as_list(lookup_path(context, spec["path"]))
    return all(value in values for value in spec["values"])


def _condition_compare(
    spec: Mapping[str, Any],
    context: Mapping[str, Any],
    compare: Any
) -> bool:
    if not path_exists(context, spec["path"]):
        return False
    return compare(lookup_path(context, spec["path"]), spec["value"])


# Condition handlers by their condition mapping key. These are the functions
# themselves rather than method names looked up with `getattr`, which is what
# the same table had to be while these lived on a class.
CONDITION_HANDLERS = {
    "all": _condition_all,
    "any": _condition_any,
    "not": _condition_not,
    "exists": _condition_exists,
    "in": _condition_in,
    "contains_any": _condition_contains_any,
    "contains_all": _condition_contains_all
}

# Handler keys whose payload is a nested condition or list of them, rather than
# a `{"path": ...}` spec. `iter_condition_paths` recurses through these.
NESTED_CONDITION_KEYS = ("all", "any", "not")


def iter_condition_paths(condition: Mapping[str, Any] | None) -> list[str]:
    """Every dotted path a condition reads, for validating authored rules.

    `evaluate_condition` treats an unresolvable path as `False`, which makes a
    misspelled path indistinguishable from a rule that correctly never fires.
    Callers that load conditions from config use this to check the paths against
    the context they intend to supply, so the mistake surfaces at load time.
    """
    if condition is None:
        return []
    paths: list[str] = []
    for key, payload in condition.items():
        if key in NESTED_CONDITION_KEYS:
            nested = payload if isinstance(payload, list) else [payload]
            for item in nested:
                paths.extend(iter_condition_paths(item))
            continue
        if isinstance(payload, Mapping) and "path" in payload:
            paths.append(payload["path"])
    return paths
