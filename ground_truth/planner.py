import re
import copy
import random
import operator
import itertools
import yaml
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Mapping, Sequence
from ground_truth.runtime import RuntimePlanCompiler
from ground_truth.operators import OperatorRegistry, load_operator_registry
from ground_truth.symbolic_graph import SymbolicEditGraph
from typing import Any

# Placeholder pattern for string interpolation in binding/spec strings.
PLACEHOLDER_RE = re.compile(r"\$\{([^}]+)\}")

# Config sources to load when building a planner bundle.
CONFIG_SOURCES = (
    ("distributions", "distributions.yaml", "distributions", {}),
    ("motifs", "motifs.yaml", "motifs", {}),
    ("recipes", "recipes.yaml", "recipes", []),
    ("constraints", "constraints.yaml", "constraints", {})
)

# Binding kinds resolved while generating plans.
BINDING_RESOLVER_NAMES = {
    "value": "_binding_constant",
    "coalesce": "_binding_coalesce",
    "ref": "_binding_ref",
    "each_from": "_binding_each_from",
    "values": "_binding_explicit_values",
    "sample": "_binding_distribution",
    "distribution": "_binding_distribution"
}

# Special expansion handlers used by `_expand_value`.
SPECIAL_VALUE_HANDLER_NAMES = {
    "coalesce": "_expand_coalesce",
    "ref": "_expand_ref",
    "sample": "_expand_sample",
    "scale": "_expand_scale",
    "tempo_sync": "_expand_tempo_sync"
}

# Condition operators available in rule/filter specs.
COMPARISON_OPERATORS = {
    "eq": operator.eq,
    "neq": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le
}

# Condition handlers for condition mapping keys.
CONDITION_HANDLER_NAMES = {
    "all": "_condition_all",
    "any": "_condition_any",
    "not": "_condition_not",
    "exists": "_condition_exists",
    "in": "_condition_in",
    "contains_any": "_condition_contains_any",
    "contains_all": "_condition_contains_all"
}

# Distribution kind handlers.
DISTRIBUTION_HANDLER_NAMES = {
    "choice": "_resolve_choice_distribution",
    "grid": "_resolve_choice_distribution",
    "values": "_resolve_choice_distribution",
    "uniform": "_resolve_uniform_distribution",
    "int_uniform": "_resolve_int_uniform_distribution"
}

# Rule action handlers.
RULE_ACTION_HANDLER_NAMES = {
    "reject": "_apply_reject_action",
    "multiply": "_apply_multiplier_action",
    "add": "_apply_additive_action"
}

# Block validation dispatch map.
BLOCK_VALIDATOR_NAMES = {
    "separate": "_validate_operator_block",
    "mix": "_validate_operator_block",
    "chain": "_validate_chain_block",
    "send_return": "_validate_chain_block",
    "step": "_validate_step_block"
}

# Default operators used when graph blocks omit operator fields.
BLOCK_DEFAULT_OPERATORS = {
    "separate": "separate_audio",
    "mix": "mix_stems"
}


@dataclass
class ConfigBundle:
    distributions: dict[str, dict[str, Any]]
    motifs: dict[str, dict[str, Any]]
    recipes: list[dict[str, Any]]
    constraints: dict[str, Any]


@dataclass
class ResolvedPlan:
    plan_id: str
    recipe_id: str
    weight: float
    bindings: dict[str, Any]
    graph_spec: list[dict[str, Any]]
    recipe_tags: list[str]
    applied_policies: list[str] = field(default_factory=list)
    poison_id: str | None = None
    poison_description: str | None = None
    poison_graph_spec: list[dict[str, Any]] | None = None
    poison_tags: list[str] = field(default_factory=list)
    poison_issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "recipe_id": self.recipe_id,
            "weight": self.weight,
            "bindings": self.bindings,
            "graph_spec": self.graph_spec,
            "recipe_tags": self.recipe_tags,
            "applied_policies": self.applied_policies,
            "poison_id": self.poison_id,
            "poison_description": self.poison_description,
            "poison_graph_spec": self.poison_graph_spec,
            "poison_tags": self.poison_tags,
            "poison_issues": self.poison_issues,
        }


class GroundTruthPlanner:
    def __init__(self, bundle: ConfigBundle, operator_registry: OperatorRegistry):
        self.bundle = bundle
        self.operator_registry = operator_registry

    @classmethod
    def from_directory(
        cls,
        config_dir: Path | str,
        operator_registry: OperatorRegistry | None = None
    ) -> "GroundTruthPlanner":
        config_root = Path(config_dir)
        bundle_data: dict[str, Any] = {}
        for field_name, file_name, root_key, default_value in CONFIG_SOURCES:
            section = cls._load_yaml(config_root / file_name).get(
                root_key,
                copy.deepcopy(default_value)
            )
            bundle_data[field_name] = section
        return cls(
            bundle=ConfigBundle(**bundle_data),
            operator_registry=operator_registry or load_operator_registry(config_root)
        )

    def plan(
        self,
        metadata: Mapping[str, Any],
        mode: str = "enumerate",
        samples_per_recipe: int = 1,
        max_variants_per_recipe: int | None = None,
        variant_selection: str = "diverse",
        seed: int | None = None
    ) -> list[ResolvedPlan]:
        if mode not in {"enumerate", "sample"}:
            raise ValueError("Unsupported planning mode '%s'." % mode)

        rng = random.Random(seed)
        plans: list[ResolvedPlan] = []
        for recipe in self.bundle.recipes:
            plans.extend(
                self._plan_recipe(
                    recipe=recipe,
                    metadata=metadata,
                    mode=mode,
                    samples_per_recipe=samples_per_recipe,
                    max_variants_per_recipe=max_variants_per_recipe,
                    variant_selection=variant_selection,
                    rng=rng
                )
            )

        plans.sort(
            key=lambda plan: (
                -plan.weight,
                plan.recipe_id,
                plan.plan_id
            )
        )
        return plans

    def random_plans(
        self,
        metadata: Mapping[str, Any],
        count: int,
        seed: int | None = None
    ) -> list[ResolvedPlan]:
        if count <= 0:
            return []

        rng = random.Random(seed)
        candidates = self._random_target_candidates(metadata)
        if not candidates:
            return []

        plans: list[ResolvedPlan] = []
        attempts = 0
        while len(plans) < count and attempts < count * 20:
            attempts += 1
            target_candidate = self._weighted_target_candidate(candidates, rng)
            plan_kind = self._weighted_random_plan_kind(target_candidate, rng)
            try:
                graph_spec, tags = self._random_graph_spec(
                    metadata=metadata,
                    target_candidate=target_candidate,
                    plan_kind=plan_kind,
                    rng=rng,
                )
            except ValueError:
                continue
            recipe = {
                "id": "random_constrained",
                "tags": tags,
                "weight": 0.5,
            }
            bindings = {
                "target_candidate": target_candidate,
                "target_description": target_candidate.get("stem"),
                "target_family": target_candidate.get("family"),
                "random_plan_kind": plan_kind,
            }
            context = self._build_context(metadata, recipe, bindings)
            allowed, weight, applied_policies = self._resolve_weight(recipe, context)
            if not allowed or weight <= 0.0:
                continue
            try:
                self.validate_graph_spec(graph_spec)
            except ValueError:
                continue
            plans.append(
                ResolvedPlan(
                    plan_id="random_constrained.%03d" % len(plans),
                    recipe_id="random_constrained",
                    weight=weight,
                    bindings=bindings,
                    graph_spec=graph_spec,
                    recipe_tags=tags,
                    applied_policies=applied_policies,
                )
            )
        return plans

    def compile_plan(self, plan: ResolvedPlan) -> SymbolicEditGraph:
        self.validate_graph_spec(plan.graph_spec)
        return RuntimePlanCompiler(self.operator_registry).compile_graph_spec(plan.graph_spec)

    def describe_plan(
        self,
        plan: ResolvedPlan,
        validate: bool = True
    ) -> str:
        return self.describe_graph_spec(plan.graph_spec, validate=validate)

    def describe_graph_spec(
        self,
        graph_spec: Sequence[Mapping[str, Any]],
        validate: bool = True
    ) -> str:
        if validate:
            self.validate_graph_spec(graph_spec)
        return SymbolicEditGraph.describe_blocks(graph_spec)

    def validate_graph_spec(self, graph_spec: Sequence[Mapping[str, Any]]) -> None:
        order_config = self.bundle.constraints.get("chain_order", {})
        for block in graph_spec:
            self._dispatch_block(
                mapping=BLOCK_VALIDATOR_NAMES,
                block=block,
                args=(block, order_config)
            )

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        return loaded or {}

    def _plan_recipe(
        self,
        recipe: Mapping[str, Any],
        metadata: Mapping[str, Any],
        mode: str,
        samples_per_recipe: int,
        max_variants_per_recipe: int | None,
        variant_selection: str,
        rng: random.Random
    ) -> list[ResolvedPlan]:
        recipe_plans: list[ResolvedPlan] = []
        variants_for_recipe = 0

        for bindings in self._expand_bindings(recipe, metadata, mode, rng):
            context = self._build_context(metadata, recipe, bindings)
            if not self._evaluate_condition(recipe.get("when"), context):
                continue

            sample_count = samples_per_recipe if mode == "sample" else 1
            for _ in range(sample_count):
                allowed, weight, applied_policies = self._resolve_weight(recipe, context)
                if not allowed or weight <= 0.0:
                    continue
                poison_variants = self._expand_recipe_poisons(
                    recipe,
                    context,
                    mode,
                    rng,
                )
                if recipe.get("poisons") and not poison_variants:
                    continue

                for graph_spec in self._expand_graph_blocks(
                    recipe.get("graph", []),
                    context,
                    mode,
                    rng
                ):
                    self.validate_graph_spec(graph_spec)
                    for poison_variant in poison_variants:
                        variants_for_recipe += 1
                        recipe_plans.append(
                            ResolvedPlan(
                                plan_id="%s.%03d" % (recipe["id"], variants_for_recipe),
                                recipe_id=recipe["id"],
                                weight=weight,
                                bindings=copy.deepcopy(bindings),
                                graph_spec=graph_spec,
                                recipe_tags=list(recipe.get("tags", [])),
                                applied_policies=applied_policies,
                                poison_id=(
                                    None if poison_variant is None else poison_variant.get("id")
                                ),
                                poison_description=(
                                    None
                                    if poison_variant is None
                                    else poison_variant.get("description")
                                ),
                                poison_graph_spec=(
                                    None
                                    if poison_variant is None
                                    else copy.deepcopy(poison_variant["graph_spec"])
                                ),
                                poison_tags=(
                                    []
                                    if poison_variant is None
                                    else list(poison_variant.get("tags", []))
                                ),
                                poison_issues=(
                                    []
                                    if poison_variant is None
                                    else list(poison_variant.get("issues", []))
                                ),
                            )
                        )

        return self._select_recipe_variants(
            recipe_plans,
            max_variants_per_recipe=max_variants_per_recipe,
            variant_selection=variant_selection,
            rng=rng,
        )

    def _expand_recipe_poisons(
        self,
        recipe: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random,
    ) -> list[dict[str, Any] | None]:
        poison_specs = list(recipe.get("poisons", []))
        if not poison_specs:
            return [None]

        expanded_poisons: list[dict[str, Any]] = []
        for poison in poison_specs:
            if not self._evaluate_condition(poison.get("when"), context):
                continue

            metadata_spec = {
                key: value
                for key, value in poison.items()
                if key not in {"graph", "when"}
            }
            metadata_variants = (
                self._expand_mapping(metadata_spec, context, mode, rng)
                if metadata_spec
                else [{}]
            )
            graph_variants = self._expand_graph_blocks(
                poison.get("graph", []),
                context,
                mode,
                rng,
            )
            for graph_spec in graph_variants:
                self.validate_graph_spec(graph_spec)
                for metadata_variant in metadata_variants:
                    expanded = dict(metadata_variant)
                    expanded["graph_spec"] = [copy.deepcopy(block) for block in graph_spec]
                    expanded_poisons.append(expanded)
        return expanded_poisons

    def _resolve_weight(
        self,
        recipe: Mapping[str, Any],
        context: Mapping[str, Any]
    ) -> tuple[bool, float, list[str]]:
        base_weight = float(recipe.get("weight", 1.0))
        allowed, weighted, recipe_policies = self._apply_rule_set(
            rules=recipe.get("weight_rules", []),
            context=context,
            base_weight=base_weight,
            action_handlers=RULE_ACTION_HANDLER_NAMES
        )
        if not allowed:
            return False, weighted, recipe_policies

        allowed, weighted, global_policies = self._apply_rule_set(
            rules=self.bundle.constraints.get("pattern_policies", []),
            context=context,
            base_weight=weighted,
            action_handlers=RULE_ACTION_HANDLER_NAMES
        )
        return allowed, weighted, recipe_policies + global_policies

    def _apply_rule_set(
        self,
        rules: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        base_weight: float,
        action_handlers: Mapping[str, str]
    ) -> tuple[bool, float, list[str]]:
        weight = base_weight
        applied: list[str] = []
        for rule in rules:
            if not self._evaluate_condition(rule.get("when"), context):
                continue
            action = rule.get("action", "multiply")
            handler_name = action_handlers.get(action)
            if handler_name is None:
                raise ValueError("Unsupported rule action '%s'." % action)
            allowed, weight = getattr(self, handler_name)(rule, weight)
            applied.append(rule.get("name", "rule"))
            if not allowed:
                return False, weight, applied
        return True, weight, applied

    @staticmethod
    def _apply_reject_action(rule: Mapping[str, Any], weight: float) -> tuple[bool, float]:
        return False, weight

    @staticmethod
    def _apply_multiplier_action(rule: Mapping[str, Any], weight: float) -> tuple[bool, float]:
        return True, weight * float(rule.get("multiplier", 1.0))

    @staticmethod
    def _apply_additive_action(rule: Mapping[str, Any], weight: float) -> tuple[bool, float]:
        return True, weight + float(rule.get("delta", 0.0))

    @staticmethod
    def _reached_variant_limit(count: int, limit: int | None) -> bool:
        return limit is not None and count >= limit

    def _select_recipe_variants(
        self,
        recipe_plans: Sequence[ResolvedPlan],
        max_variants_per_recipe: int | None,
        variant_selection: str,
        rng: random.Random,
    ) -> list[ResolvedPlan]:
        if max_variants_per_recipe is None or len(recipe_plans) <= max_variants_per_recipe:
            return list(recipe_plans)
        if variant_selection == "random":
            selected = rng.sample(list(recipe_plans), max_variants_per_recipe)
        elif variant_selection == "diverse":
            selected = self._select_diverse_recipe_variants(
                recipe_plans=recipe_plans,
                limit=max_variants_per_recipe,
                rng=rng,
            )
        else:
            raise ValueError("Unsupported variant_selection '%s'." % variant_selection)

        selected = sorted(selected, key=lambda plan: (-plan.weight, plan.plan_id))
        return [
            ResolvedPlan(
                plan_id="%s.%03d" % (plan.recipe_id, index),
                recipe_id=plan.recipe_id,
                weight=plan.weight,
                bindings=copy.deepcopy(plan.bindings),
                graph_spec=copy.deepcopy(plan.graph_spec),
                recipe_tags=list(plan.recipe_tags),
                applied_policies=list(plan.applied_policies),
                poison_id=plan.poison_id,
                poison_description=plan.poison_description,
                poison_graph_spec=(
                    None if plan.poison_graph_spec is None else copy.deepcopy(plan.poison_graph_spec)
                ),
                poison_tags=list(plan.poison_tags),
                poison_issues=list(plan.poison_issues),
            )
            for index, plan in enumerate(selected, start=1)
        ]

    def _select_diverse_recipe_variants(
        self,
        recipe_plans: Sequence[ResolvedPlan],
        limit: int,
        rng: random.Random,
    ) -> list[ResolvedPlan]:
        remaining = list(recipe_plans)
        rng.shuffle(remaining)
        selected: list[ResolvedPlan] = []
        while remaining and len(selected) < limit:
            best = max(
                remaining,
                key=lambda plan: (
                    self._recipe_variant_extremeness(plan, recipe_plans),
                    self._min_variant_distance(plan, selected, recipe_plans),
                    plan.weight,
                    rng.random(),
                ),
            )
            selected.append(best)
            remaining = [plan for plan in remaining if plan.plan_id != best.plan_id]
        return selected

    def _min_variant_distance(
        self,
        plan: ResolvedPlan,
        selected: Sequence[ResolvedPlan],
        family: Sequence[ResolvedPlan],
    ) -> float:
        if not selected:
            return 1.0
        return min(self._variant_distance(plan, other, family) for other in selected)

    def _variant_distance(
        self,
        left: ResolvedPlan,
        right: ResolvedPlan,
        family: Sequence[ResolvedPlan],
    ) -> float:
        left_params = self._flatten_plan_numeric_params(left)
        right_params = self._flatten_plan_numeric_params(right)
        keys = sorted(set(left_params) | set(right_params))
        if not keys:
            return 0.0
        family_params = [self._flatten_plan_numeric_params(plan) for plan in family]
        distances: list[float] = []
        for key in keys:
            values = [params[key] for params in family_params if key in params]
            if not values:
                continue
            scale = max(max(values) - min(values), 1e-9)
            distances.append(abs(left_params.get(key, min(values)) - right_params.get(key, min(values))) / scale)
        return sum(distances) / len(distances) if distances else 0.0

    def _recipe_variant_extremeness(
        self,
        plan: ResolvedPlan,
        family: Sequence[ResolvedPlan],
    ) -> float:
        params = self._flatten_plan_numeric_params(plan)
        if not params:
            return 0.0
        family_params = [self._flatten_plan_numeric_params(item) for item in family]
        scores: list[float] = []
        for key, value in params.items():
            values = [item[key] for item in family_params if key in item]
            if len(values) < 2:
                continue
            low = min(values)
            high = max(values)
            if low == high:
                continue
            midpoint = (low + high) / 2.0
            half_range = (high - low) / 2.0
            scores.append(min(abs(value - midpoint) / half_range, 1.0))
        return sum(scores) / len(scores) if scores else 0.0

    def _flatten_plan_numeric_params(self, plan: ResolvedPlan) -> dict[str, float]:
        flattened = {
            "final:%s" % key: value
            for key, value in self._flatten_numeric_params(plan.graph_spec).items()
        }
        if plan.poison_graph_spec:
            flattened.update(
                {
                    "poison:%s" % key: value
                    for key, value in self._flatten_numeric_params(plan.poison_graph_spec).items()
                }
            )
        return flattened

    @staticmethod
    def _flatten_numeric_params(graph_spec: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        flattened: dict[str, float] = {}
        for block in graph_spec:
            if block.get("kind") == "send_return":
                for key in ("dry_level", "send_level", "return_level"):
                    value = block.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        flattened["%s:%s" % (block.get("name"), key)] = float(value)
            for step in block.get("steps", []):
                operator_name = step.get("operator")
                for param_name, param_value in step.get("params", {}).items():
                    if isinstance(param_value, (int, float)) and not isinstance(param_value, bool):
                        flattened["%s:%s" % (operator_name, param_name)] = float(param_value)
        return flattened

    def _build_context(
        self,
        metadata: Mapping[str, Any],
        recipe: Mapping[str, Any],
        bindings: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "metadata": dict(metadata),
            "bindings": dict(bindings),
            "pattern": {
                "id": recipe["id"],
                "tags": list(recipe.get("tags", [])),
                "weight": float(recipe.get("weight", 1.0))
            }
        }

    def _random_target_candidates(self, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
        analysis = dict(metadata.get("analysis", {}))
        candidates: list[dict[str, Any]] = []
        for candidate in analysis.get("target_candidates", []):
            if not isinstance(candidate, Mapping):
                continue
            family = candidate.get("family")
            stem = candidate.get("stem")
            if family in (None, "other") or stem in (None, ""):
                continue
            candidates.append(dict(candidate))
        return candidates

    def _weighted_target_candidate(
        self,
        candidates: Sequence[Mapping[str, Any]],
        rng: random.Random,
    ) -> dict[str, Any]:
        weights = [
            self._target_family_random_weight(str(candidate.get("family")))
            for candidate in candidates
        ]
        return dict(rng.choices(list(candidates), weights=weights, k=1)[0])

    @staticmethod
    def _target_family_random_weight(family: str) -> float:
        return {
            "vocals": 3.0,
            "drums": 3.0,
            "bass": 1.6,
            "guitar": 1.4,
            "piano": 1.1,
            "keys": 1.1,
        }.get(family, 1.0)

    def _weighted_random_plan_kind(
        self,
        target_candidate: Mapping[str, Any],
        rng: random.Random,
    ) -> str:
        return "random_chain"

    def _random_graph_spec(
        self,
        metadata: Mapping[str, Any],
        target_candidate: Mapping[str, Any],
        plan_kind: str,
        rng: random.Random,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        target_description = str(target_candidate["stem"])
        family = str(target_candidate.get("family"))
        topology = self._random_topology(family, rng)
        serial_max_steps = 7 if topology == "serial_plus_send" else 10
        serial_steps, serial_tags = self._random_steps(
            metadata,
            target_candidate,
            plan_kind,
            rng,
            forbidden_tags=["time_based"] if topology == "serial_plus_send" else [],
            min_steps=3,
            max_steps=serial_max_steps,
        )
        send_steps: list[dict[str, Any]] = []
        send_tags: list[str] = []
        if topology in {"send_return", "serial_plus_send"}:
            max_send_steps = 10 if topology == "send_return" else max(3, 10 - len(serial_steps))
            send_steps, send_tags = self._random_send_steps(metadata, family, rng, max_send_steps=max_send_steps)

        blocks: list[dict[str, Any]] = [
            {
                "kind": "separate",
                "name": "isolate_random_target",
                "source": "audio",
                "description": target_description,
                "outputs": [
                    "target_stem",
                    "residual",
                ],
            },
        ]
        if topology == "serial":
            self._validate_random_chain(serial_steps, family, "serial")
            blocks.append(
                {
                    "kind": "chain",
                    "prefix": "random_fx",
                    "order_profile": "shaping",
                    "source": "target_stem",
                    "output": "processed_stem",
                    "steps": serial_steps,
                }
            )
        elif topology == "send_return":
            self._validate_random_chain(send_steps, family, "send_return")
            blocks.append(
                {
                    "kind": "send_return",
                    "name": "random_send",
                    "order_profile": "send_texture",
                    "source": "target_stem",
                    "output": "processed_stem",
                    "dry_level": 1.0,
                    "send_level": rng.choice([0.25, 0.4, 0.6]),
                    "return_level": rng.choice([0.45, 0.7, 0.9]),
                    "steps": send_steps,
                }
            )
        else:
            self._validate_random_chain(serial_steps, family, "serial")
            self._validate_random_chain(send_steps, family, "send_return")
            blocks.append(
                {
                    "kind": "chain",
                    "prefix": "random_serial",
                    "order_profile": "shaping",
                    "source": "target_stem",
                    "output": "serial_stem",
                    "steps": serial_steps,
                }
            )
            blocks.append(
                {
                    "kind": "send_return",
                    "name": "random_send",
                    "order_profile": "send_texture",
                    "source": "serial_stem",
                    "output": "processed_stem",
                    "dry_level": 1.0,
                    "send_level": rng.choice([0.2, 0.35, 0.5]),
                    "return_level": rng.choice([0.4, 0.65, 0.85]),
                    "steps": send_steps,
                }
            )
        blocks.append(
            {
                "kind": "mix",
                "name": "remix_random_target",
                "stem": "processed_stem",
                "residual": "residual",
                "output": "final_audio",
            },
        )
        tags = self._unique_list(["random", "constrained", topology] + serial_tags + send_tags)
        return blocks, tags

    @staticmethod
    def _random_order_profile(tags: Sequence[str]) -> str:
        return "shaping"

    def _random_steps(
        self,
        metadata: Mapping[str, Any],
        target_candidate: Mapping[str, Any],
        plan_kind: str,
        rng: random.Random,
        forbidden_tags: Sequence[str],
        min_steps: int,
        max_steps: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        operator_names = self._sample_random_operator_names(
            str(target_candidate.get("family")),
            rng,
            forbidden_tags,
            min_steps=min_steps,
            max_steps=max_steps,
        )
        steps = [
            self._random_step_for_operator(
                operator_name=operator_name,
                metadata=metadata,
                target_candidate=target_candidate,
                rng=rng,
                index=index,
            )
            for index, operator_name in enumerate(operator_names, start=1)
        ]
        steps = self._sort_random_steps_by_precedence(steps, "shaping")
        tags = self._operator_tags_for_steps(steps)
        return steps, tags

    def _random_topology(
        self,
        family: str,
        rng: random.Random,
    ) -> str:
        if family == "drums":
            values = ["serial", "serial_plus_send"]
            weights = [4.0, 1.0]
        else:
            values = ["serial", "send_return", "serial_plus_send"]
            weights = [3.0, 2.0, 3.0]
        return str(rng.choices(values, weights=weights, k=1)[0])

    def _sample_random_operator_names(
        self,
        family: str,
        rng: random.Random,
        forbidden_tags: Sequence[str],
        min_steps: int,
        max_steps: int,
    ) -> list[str]:
        pool = [
            item
            for item in self._random_operator_pool(family)
            if not self._operator_has_any_tag(str(item["operator"]), forbidden_tags)
        ]
        length = rng.randint(min_steps, min(max_steps, len(pool)))
        selected: list[str] = []
        remaining = list(pool)
        while len(selected) < length and remaining:
            values = [str(item["operator"]) for item in remaining]
            weights = [float(item.get("weight", 1.0)) for item in remaining]
            chosen = str(rng.choices(values, weights=weights, k=1)[0])
            selected.append(chosen)
            remaining = [item for item in remaining if item["operator"] != chosen]
        return selected

    @staticmethod
    def _random_operator_pool(family: str) -> list[dict[str, Any]]:
        common = [
            {"operator": "apply_peak_filter", "weight": 1.0},
            {"operator": "apply_highshelf_filter", "weight": 1.0},
            {"operator": "apply_lowshelf_filter", "weight": 0.8},
            {"operator": "apply_highpass_filter", "weight": 1.0},
            {"operator": "apply_lowpass_filter", "weight": 1.0},
            {"operator": "apply_compressor_effect", "weight": 1.4},
            {"operator": "apply_limiter_effect", "weight": 0.7},
            {"operator": "apply_gain", "weight": 0.9},
        ]
        color = [
            {"operator": "apply_distortion_effect", "weight": 0.9},
            {"operator": "apply_chorus_effect", "weight": 0.8},
            {"operator": "apply_phaser_effect", "weight": 0.6},
            {"operator": "apply_delay_effect", "weight": 0.9},
            {"operator": "apply_reverb_effect", "weight": 0.9},
        ]
        if family == "drums":
            return common + [
                {"operator": "apply_distortion_effect", "weight": 1.5},
                {"operator": "apply_chorus_effect", "weight": 0.25},
                {"operator": "apply_phaser_effect", "weight": 0.2},
                {"operator": "apply_reverb_effect", "weight": 0.4},
            ]
        return common + color

    def _random_send_steps(
        self,
        metadata: Mapping[str, Any],
        family: str,
        rng: random.Random,
        max_send_steps: int,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        pool = [
            {"operator": "apply_chorus_effect", "weight": 1.0},
            {"operator": "apply_phaser_effect", "weight": 0.8},
            {"operator": "apply_delay_effect", "weight": 1.4},
            {"operator": "apply_reverb_effect", "weight": 1.4},
            {"operator": "apply_highpass_filter", "weight": 1.0},
            {"operator": "apply_lowpass_filter", "weight": 1.0},
            {"operator": "apply_gain", "weight": 0.5},
        ]
        if family == "drums":
            pool = [
                item
                for item in pool
                if item["operator"] not in {"apply_delay_effect", "apply_chorus_effect", "apply_phaser_effect"}
            ]
        length = rng.randint(3, min(max_send_steps, len(pool)))
        selected: list[str] = []
        remaining = list(pool)
        while len(selected) < length and remaining:
            values = [str(item["operator"]) for item in remaining]
            weights = [float(item.get("weight", 1.0)) for item in remaining]
            chosen = str(rng.choices(values, weights=weights, k=1)[0])
            selected.append(chosen)
            remaining = [item for item in remaining if item["operator"] != chosen]
        steps = [
            self._random_step_for_operator(
                operator_name=operator_name,
                metadata=metadata,
                target_candidate={"family": family},
                rng=rng,
                index=index,
            )
            for index, operator_name in enumerate(selected, start=1)
        ]
        steps = self._sort_random_steps_by_precedence(steps, "send_texture")
        return steps, self._operator_tags_for_steps(steps)

    def _random_step_for_operator(
        self,
        operator_name: str,
        metadata: Mapping[str, Any],
        target_candidate: Mapping[str, Any],
        rng: random.Random,
        index: int,
    ) -> dict[str, Any]:
        name = "%02d_%s" % (index, operator_name.replace("apply_", "").replace("_effect", ""))
        if operator_name == "apply_peak_filter":
            return self._peak_step(rng, name)
        if operator_name == "apply_highshelf_filter":
            step = self._retro_shelf_step(rng)
            step["name"] = name
            return step
        if operator_name == "apply_lowshelf_filter":
            return self._lowshelf_step(rng, name)
        if operator_name == "apply_highpass_filter":
            return self._eq_highpass_step(rng, name)
        if operator_name == "apply_lowpass_filter":
            return self._eq_lowpass_step(rng, name)
        if operator_name == "apply_compressor_effect":
            return self._compressor_step(target_candidate, rng, name)
        if operator_name == "apply_limiter_effect":
            return self._limiter_step(rng, name)
        if operator_name == "apply_gain":
            return self._gain_step(rng, name)
        if operator_name == "apply_distortion_effect":
            return self._distortion_step(rng, name)
        if operator_name == "apply_chorus_effect":
            step = self._chorus_step(rng)
            step["name"] = name
            return step
        if operator_name == "apply_phaser_effect":
            return self._phaser_step(rng, name)
        if operator_name == "apply_delay_effect":
            step = self._delay_step(metadata, rng)
            step["name"] = name
            return step
        if operator_name == "apply_reverb_effect":
            step = self._reverb_step(rng)
            step["name"] = name
            return step
        raise ValueError("Unsupported random operator '%s'." % operator_name)

    def _sort_random_steps_by_precedence(
        self,
        steps: Sequence[Mapping[str, Any]],
        profile_name: str,
    ) -> list[dict[str, Any]]:
        profile = self.bundle.constraints.get("chain_order", {}).get("profiles", {}).get(profile_name, {})
        tag_order = profile.get("tags", profile)
        indexed_steps = list(enumerate(steps))
        indexed_steps.sort(
            key=lambda item: (
                self._step_precedence_rank(item[1], tag_order),
                item[0],
            )
        )
        return [dict(step) for _, step in indexed_steps]

    def _step_precedence_rank(
        self,
        step: Mapping[str, Any],
        tag_order: Mapping[str, int],
    ) -> int:
        operator = self.operator_registry.resolve(str(step["operator"]))
        ranks = [
            int(tag_order[tag])
            for tag in operator.tags
            if tag in tag_order
        ]
        return min(ranks) if ranks else 10_000

    def _operator_tags_for_steps(self, steps: Sequence[Mapping[str, Any]]) -> list[str]:
        tags: list[str] = []
        for step in steps:
            operator = self.operator_registry.resolve(str(step["operator"]))
            for tag in operator.tags:
                if tag not in tags:
                    tags.append(tag)
        return tags

    def _operator_has_any_tag(
        self,
        operator_name: str,
        tags: Sequence[str],
    ) -> bool:
        if not tags:
            return False
        operator = self.operator_registry.resolve(operator_name)
        return any(tag in operator.tags for tag in tags)

    @staticmethod
    def _unique_list(values: Sequence[str]) -> list[str]:
        unique: list[str] = []
        for value in values:
            if value not in unique:
                unique.append(value)
        return unique

    def _validate_random_chain(
        self,
        steps: Sequence[Mapping[str, Any]],
        family: str,
        topology: str,
    ) -> None:
        if len(steps) < 3 or len(steps) > 10:
            raise ValueError("Random chain length must be in [3, 10].")
        counts: dict[str, int] = {}
        highpass_hz = 0.0
        lowpass_hz = 24_000.0
        gain_budget = 0.0
        for step in steps:
            operator_name = str(step["operator"])
            operator = self.operator_registry.resolve(operator_name)
            for tag in operator.tags:
                counts[tag] = counts.get(tag, 0) + 1
            params = dict(step.get("params", {}))
            if operator_name == "apply_highpass_filter":
                highpass_hz = max(highpass_hz, float(params["cutoff_frequency_hz"]))
            if operator_name == "apply_lowpass_filter":
                lowpass_hz = min(lowpass_hz, float(params["cutoff_frequency_hz"]))
            if operator_name == "apply_gain":
                gain_budget += float(params["gain_db"])
            if operator_name in {"apply_highshelf_filter", "apply_lowshelf_filter", "apply_peak_filter"}:
                gain_budget += 0.5 * float(params["gain_db"])

        if highpass_hz > 0.0 and lowpass_hz < 24_000.0:
            if highpass_hz >= lowpass_hz * 0.6:
                raise ValueError("Random chain spectral passband is too narrow.")
            if lowpass_hz / max(highpass_hz, 1.0) < 8.0:
                raise ValueError("Random chain removes too much spectrum.")
        if topology == "serial":
            if highpass_hz > self._max_main_highpass_hz(family):
                raise ValueError("Random chain highpass is too aggressive for main path.")
            if lowpass_hz < self._min_main_lowpass_hz(family):
                raise ValueError("Random chain lowpass is too aggressive for main path.")
        if abs(gain_budget) > 10.0:
            raise ValueError("Random chain cumulative gain budget exceeds +/-10 dB.")
        if counts.get("dynamics", 0) > 2:
            raise ValueError("Random chain has too many dynamics operators.")
        if counts.get("distortion", 0) > 1:
            raise ValueError("Random chain has too many distortion operators.")
        if counts.get("modulation", 0) > 1:
            raise ValueError("Random chain has too many modulation operators.")
        if counts.get("time_based", 0) > 2:
            raise ValueError("Random chain has too many time-based operators.")
        if counts.get("eq_pass", 0) > 2:
            raise ValueError("Random chain has too many pass filters.")
        if counts.get("gain", 0) > 1:
            raise ValueError("Random chain has too many gain operators.")
        if family == "drums" and counts.get("delay", 0) > 0:
            raise ValueError("Random chain placed delay on drums.")
        if family == "bass" and (counts.get("time_based", 0) > 0 or counts.get("modulation", 0) > 0):
            raise ValueError("Random chain placed time/modulation on bass.")

    @staticmethod
    def _max_main_highpass_hz(family: str) -> float:
        return {
            "bass": 120.0,
            "drums": 180.0,
            "vocals": 350.0,
        }.get(family, 300.0)

    @staticmethod
    def _min_main_lowpass_hz(family: str) -> float:
        return {
            "bass": 4500.0,
            "drums": 4500.0,
            "vocals": 4500.0,
        }.get(family, 4500.0)

    def _compressor_step(
        self,
        target_candidate: Mapping[str, Any],
        rng: random.Random,
        name: str,
    ) -> dict[str, Any]:
        family = str(target_candidate.get("family"))
        threshold_distribution = "drums.parallel_compression.threshold_db" if family == "drums" else "vocals.compression.threshold_db"
        ratio_distribution = "drums.parallel_compression.ratio" if family == "drums" else "vocals.compression.ratio"
        return {
            "name": name,
            "operator": "apply_compressor_effect",
            "params": {
                "threshold_db": self._sample_distribution(threshold_distribution, rng),
                "ratio": self._sample_distribution(ratio_distribution, rng),
                "attack_ms": self._sample_distribution("vocals.compression.attack_ms", rng),
                "release_ms": self._sample_distribution("vocals.compression.release_ms", rng),
            },
        }

    def _delay_step(
        self,
        metadata: Mapping[str, Any],
        rng: random.Random,
    ) -> dict[str, Any]:
        return {
            "name": "delay",
            "operator": "apply_delay_effect",
            "params": {
                "delay_seconds": self._random_delay_seconds(metadata, rng),
                "feedback": self._sample_distribution("space.shared_send.feedback", rng),
                "mix": 0.45,
            },
        }

    def _reverb_step(self, rng: random.Random) -> dict[str, Any]:
        return {
            "name": "reverb",
            "operator": "apply_reverb_effect",
            "params": {
                "room_size": self._sample_distribution("space.shared_send.reverb.room_size", rng),
                "damping": self._sample_distribution("space.shared_send.reverb.damping", rng),
                "wet_level": 0.55,
                "dry_level": 0.65,
                "width": 1.0,
                "freeze_mode": 0.0,
            },
        }

    def _retro_shelf_step(self, rng: random.Random) -> dict[str, Any]:
        return {
            "name": "retro_shelf",
            "operator": "apply_highshelf_filter",
            "params": {
                "cutoff_frequency_hz": self._sample_distribution("tone.retro.shelf_cut_hz", rng),
                "gain_db": self._sample_distribution("tone.retro.shelf_gain_db", rng),
                "q": 0.7071067690849304,
            },
        }

    def _chorus_step(self, rng: random.Random) -> dict[str, Any]:
        return {
            "name": "chorus",
            "operator": "apply_chorus_effect",
            "params": {
                "rate_hz": self._sample_distribution("tone.modulation.chorus.rate_hz", rng),
                "depth": self._sample_distribution("tone.modulation.chorus.depth", rng),
                "centre_delay_ms": 7.0,
                "feedback": 0.0,
                "mix": self._sample_distribution("tone.modulation.chorus.mix", rng),
            },
        }

    def _gain_step(
        self,
        rng: random.Random,
        name: str,
        force_down: bool = False,
    ) -> dict[str, Any]:
        gain_db = (
            self._sample_distribution("balance.target_gain.down_db", rng)
            if force_down
            else rng.choice(
                [
                    self._sample_distribution("balance.target_gain.up_db", rng),
                    self._sample_distribution("balance.target_gain.down_db", rng),
                ]
            )
        )
        return {
            "name": name,
            "operator": "apply_gain",
            "params": {
                "gain_db": gain_db,
            },
        }

    def _eq_highpass_step(self, rng: random.Random, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "operator": "apply_highpass_filter",
            "params": {
                "cutoff_frequency_hz": rng.choice([80.0, 120.0, 180.0, 300.0]),
            },
        }

    def _eq_lowpass_step(self, rng: random.Random, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "operator": "apply_lowpass_filter",
            "params": {
                "cutoff_frequency_hz": rng.choice([4500.0, 6500.0, 8500.0, 12000.0]),
            },
        }

    def _peak_step(self, rng: random.Random, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "operator": "apply_peak_filter",
            "params": {
                "cutoff_frequency_hz": rng.choice([350.0, 800.0, 1800.0, 3200.0, 5200.0]),
                "gain_db": rng.choice([-6.0, -3.0, 3.0, 5.0]),
                "q": rng.choice([0.8, 1.5, 3.0]),
            },
        }

    def _lowshelf_step(self, rng: random.Random, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "operator": "apply_lowshelf_filter",
            "params": {
                "cutoff_frequency_hz": rng.choice([90.0, 140.0, 220.0]),
                "gain_db": rng.choice([-5.0, -3.0, 3.0, 5.0]),
                "q": 0.7071067690849304,
            },
        }

    def _limiter_step(self, rng: random.Random, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "operator": "apply_limiter_effect",
            "params": {
                "threshold_db": rng.choice([-10.0, -6.0, -3.0]),
                "release_ms": rng.choice([60.0, 120.0, 220.0]),
            },
        }

    def _distortion_step(self, rng: random.Random, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "operator": "apply_distortion_effect",
            "params": {
                "drive_db": rng.choice([5.0, 8.0, 12.0, 16.0]),
            },
        }

    def _phaser_step(self, rng: random.Random, name: str) -> dict[str, Any]:
        return {
            "name": name,
            "operator": "apply_phaser_effect",
            "params": {
                "rate_hz": rng.choice([0.25, 0.5, 1.0]),
                "depth": rng.choice([0.35, 0.55, 0.75]),
                "centre_frequency_hz": rng.choice([650.0, 1300.0, 2200.0]),
                "feedback": rng.choice([0.0, 0.15, 0.3]),
                "mix": rng.choice([0.25, 0.4, 0.6]),
            },
        }

    def _random_delay_seconds(
        self,
        metadata: Mapping[str, Any],
        rng: random.Random,
    ) -> float:
        analysis = dict(metadata.get("analysis", {}))
        bpm = analysis.get("tempo_bpm")
        if bpm is not None and rng.random() < 0.7:
            beats = float(self._sample_distribution("space.shared_send.delay.synced.beats", rng))
            return (60.0 / float(bpm)) * beats
        return float(self._sample_distribution("space.shared_send.delay.free.seconds", rng))

    def _sample_distribution(self, distribution_name: str, rng: random.Random) -> Any:
        return self._distribution_values(distribution_name, "sample", rng)[0]

    def _expand_bindings(
        self,
        recipe: Mapping[str, Any],
        metadata: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[dict[str, Any]]:
        binding_specs = recipe.get("bindings", {})
        if not binding_specs:
            return [{}]
        expanded_bindings: list[dict[str, Any]] = [{}]
        for name, spec in binding_specs.items():
            next_bindings: list[dict[str, Any]] = []
            for partial in expanded_bindings:
                context = {"metadata": dict(metadata), "bindings": dict(partial)}
                for value in self._binding_values(spec, context, mode, rng):
                    updated = dict(partial)
                    updated[name] = value
                    next_bindings.append(updated)
            expanded_bindings = next_bindings
        return expanded_bindings

    def _binding_values(
        self,
        spec: Any,
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[Any]:
        if not isinstance(spec, Mapping):
            return [spec]

        key = next((key for key in BINDING_RESOLVER_NAMES if key in spec), None)
        if key is None:
            raise ValueError("Unsupported binding spec '%s'." % spec)
        handler_name = BINDING_RESOLVER_NAMES[key]
        return getattr(self, handler_name)(spec, context, mode, rng, key)

    @staticmethod
    def _binding_constant(
        spec: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random,
        key: str
    ) -> list[Any]:
        return [spec[key]]

    def _binding_coalesce(
        self,
        spec: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random,
        key: str
    ) -> list[Any]:
        return self._expand_coalesce(spec[key], context, mode, rng)

    def _binding_ref(
        self,
        spec: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random,
        key: str
    ) -> list[Any]:
        return [self._lookup_path(context, spec[key])]

    def _binding_each_from(
        self,
        spec: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random,
        key: str
    ) -> list[Any]:
        values = list(self._lookup_path(context, spec[key]))
        values = self._filter_binding_values(values, spec)
        return [rng.choice(values)] if mode == "sample" else values

    def _binding_explicit_values(
        self,
        spec: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random,
        key: str
    ) -> list[Any]:
        values = self._filter_binding_values(list(spec[key]), spec)
        return [rng.choice(values)] if mode == "sample" else values

    def _binding_distribution(
        self,
        spec: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random,
        key: str
    ) -> list[Any]:
        return self._distribution_values(spec[key], mode, rng)

    @staticmethod
    def _filter_binding_values(values: list[Any], spec: Mapping[str, Any]) -> list[Any]:
        include = set(spec.get("include", []))
        exclude = set(spec.get("exclude", []))
        if not include and not exclude:
            return list(values)
        filtered = [
            value
            for value in values
            if (not include or value in include) and value not in exclude
        ]
        if not filtered:
            raise ValueError("Binding filter removed all candidate values.")
        return filtered

    def _expand_graph_blocks(
        self,
        blocks: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[list[dict[str, Any]]]:
        block_variants = [
            self._expand_value(self._normalize_block(block), context, mode, rng)
            for block in blocks
        ]
        return [
            [copy.deepcopy(item) for item in combination]
            for combination in itertools.product(*block_variants)
        ]

    def _normalize_block(self, block: Mapping[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(dict(block))
        if "chain_ref" in normalized:
            motif = copy.deepcopy(self.bundle.motifs[normalized.pop("chain_ref")])
            for key, value in motif.items():
                normalized.setdefault(key, value)
        return normalized

    def _expand_value(
        self,
        value: Any,
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[Any]:
        if isinstance(value, dict):
            special_key = self._special_value_key(value)
            if special_key is not None:
                handler_name = SPECIAL_VALUE_HANDLER_NAMES[special_key]
                return getattr(self, handler_name)(value[special_key], context, mode, rng)
            return self._expand_mapping(value, context, mode, rng)

        if isinstance(value, list):
            item_values = [
                self._expand_value(item, context, mode, rng)
                for item in value
            ]
            return [list(combination) for combination in itertools.product(*item_values)]

        if isinstance(value, str):
            return [self._interpolate_string(value, context)]
        return [value]

    @staticmethod
    def _special_value_key(value: Mapping[str, Any]) -> str | None:
        if len(value) != 1:
            return None
        key = next(iter(value.keys()))
        if key not in SPECIAL_VALUE_HANDLER_NAMES:
            return None
        return key

    def _expand_mapping(
        self,
        value: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[dict[str, Any]]:
        item_keys = list(value.keys())
        item_values = [self._expand_value(value[key], context, mode, rng) for key in item_keys]
        return [
            {
                item_keys[index]: combination[index]
                for index in range(len(item_keys))
            }
            for combination in itertools.product(*item_values)
        ]

    def _expand_coalesce(
        self,
        candidates: Sequence[Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[Any]:
        for candidate in candidates:
            if self._can_resolve(candidate, context):
                return self._expand_value(candidate, context, mode, rng)
        raise ValueError("Coalesce did not find a resolvable candidate.")

    def _expand_ref(self, path: str, context: Mapping[str, Any], mode: str, rng: random.Random) -> list[Any]:
        return [self._lookup_path(context, path)]

    def _expand_sample(
        self,
        distribution_name: str,
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[Any]:
        return self._distribution_values(distribution_name, mode, rng)

    def _expand_scale(
        self,
        spec: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random,
    ) -> list[float]:
        factor = float(spec.get("factor", 1.0))
        offset = float(spec.get("offset", 0.0))
        values = self._expand_value(spec["value"], context, mode, rng)
        return [(float(value) * factor) + offset for value in values]

    def _expand_tempo_sync(
        self,
        spec: Mapping[str, Any],
        context: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[float]:
        bpm_values = [
            value for value in self._expand_value(spec["bpm"], context, mode, rng)
            if value is not None
        ]
        beat_values = self._expand_value(spec["beats"], context, mode, rng)
        return [
            (60.0 / float(bpm)) * float(beats)
            for bpm, beats in itertools.product(bpm_values, beat_values)
        ]

    def _distribution_values(
        self,
        distribution_name: str,
        mode: str,
        rng: random.Random
    ) -> list[Any]:
        spec = self._distribution_spec(distribution_name)
        distribution_type = spec.get("type", "choice")
        handler_name = DISTRIBUTION_HANDLER_NAMES.get(distribution_type)
        if handler_name is None:
            raise ValueError(
                "Unsupported distribution type '%s' for '%s'." % (
                    distribution_type,
                    distribution_name
                )
            )
        return getattr(self, handler_name)(spec, mode, rng)

    def _resolve_choice_distribution(
        self,
        spec: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[Any]:
        values, weights = self._choice_values(spec)
        if mode == "sample":
            return [rng.choices(values, weights=weights, k=1)[0]]
        return values

    def _resolve_uniform_distribution(
        self,
        spec: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[float]:
        if mode == "sample":
            return [rng.uniform(float(spec["low"]), float(spec["high"]))]
        return self._enumerate_uniform(spec)

    def _resolve_int_uniform_distribution(
        self,
        spec: Mapping[str, Any],
        mode: str,
        rng: random.Random
    ) -> list[Any]:
        if mode == "sample":
            return [rng.randint(int(spec["low"]), int(spec["high"]))]
        return self._enumerate_int_uniform(spec)

    @staticmethod
    def _choice_values(spec: Mapping[str, Any]) -> tuple[list[Any], list[float]]:
        values: list[Any] = []
        weights: list[float] = []
        for item in spec.get("values", []):
            if isinstance(item, Mapping) and "value" in item:
                values.append(item["value"])
                weights.append(float(item.get("weight", 1.0)))
                continue
            values.append(item)
            weights.append(1.0)
        if not values:
            raise ValueError("Choice distribution requires at least one value.")
        return values, weights

    @staticmethod
    def _enumerate_uniform(spec: Mapping[str, Any]) -> list[float]:
        if "samples" in spec:
            return [float(value) for value in spec["samples"]]
        count = int(spec.get("count", 3))
        low = float(spec["low"])
        high = float(spec["high"])
        if count == 1:
            return [low]
        step = (high - low) / float(count - 1)
        return [low + (step * index) for index in range(count)]

    @staticmethod
    def _enumerate_int_uniform(spec: Mapping[str, Any]) -> list[int]:
        if "samples" in spec:
            return [int(value) for value in spec["samples"]]
        low = int(spec["low"])
        high = int(spec["high"])
        return list(range(low, high + 1))

    def _distribution_spec(self, distribution_name: str) -> Mapping[str, Any]:
        current: Any = self.bundle.distributions
        for segment in distribution_name.split("."):
            if not isinstance(current, Mapping) or segment not in current:
                raise ValueError("Unknown distribution '%s'." % distribution_name)
            current = current[segment]
        if not isinstance(current, Mapping):
            raise ValueError("Distribution '%s' did not resolve to a mapping." % distribution_name)
        if not self._looks_like_distribution(current):
            raise ValueError(
                "Distribution '%s' resolved to a grouping node, not a leaf distribution." % (
                    distribution_name
                )
            )
        return current

    @staticmethod
    def _looks_like_distribution(spec: Mapping[str, Any]) -> bool:
        return "type" in spec or "values" in spec or ("low" in spec and "high" in spec)

    def _evaluate_condition(
        self,
        condition: Mapping[str, Any] | None,
        context: Mapping[str, Any]
    ) -> bool:
        if condition is None:
            return True
        for key, compare in COMPARISON_OPERATORS.items():
            if key in condition:
                return self._condition_compare(condition[key], context, compare)
        for key, handler_name in CONDITION_HANDLER_NAMES.items():
            if key in condition:
                return getattr(self, handler_name)(condition[key], context)
        raise ValueError("Unsupported condition '%s'." % condition)

    def _condition_all(self, conditions: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> bool:
        return all(self._evaluate_condition(item, context) for item in conditions)

    def _condition_any(self, conditions: Sequence[Mapping[str, Any]], context: Mapping[str, Any]) -> bool:
        return any(self._evaluate_condition(item, context) for item in conditions)

    def _condition_not(self, condition: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        return not self._evaluate_condition(condition, context)

    def _condition_exists(self, spec: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        return self._path_exists(context, spec["path"])

    def _condition_in(self, spec: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        if not self._path_exists(context, spec["path"]):
            return False
        return self._lookup_path(context, spec["path"]) in list(spec["values"])

    def _condition_contains_any(self, spec: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        if not self._path_exists(context, spec["path"]):
            return False
        values = self._as_list(self._lookup_path(context, spec["path"]))
        return any(value in values for value in spec["values"])

    def _condition_contains_all(self, spec: Mapping[str, Any], context: Mapping[str, Any]) -> bool:
        if not self._path_exists(context, spec["path"]):
            return False
        values = self._as_list(self._lookup_path(context, spec["path"]))
        return all(value in values for value in spec["values"])

    def _condition_compare(
        self,
        spec: Mapping[str, Any],
        context: Mapping[str, Any],
        compare: Any
    ) -> bool:
        if not self._path_exists(context, spec["path"]):
            return False
        return compare(self._lookup_path(context, spec["path"]), spec["value"])

    def _can_resolve(self, value: Any, context: Mapping[str, Any]) -> bool:
        if value is None:
            return False
        if isinstance(value, Mapping):
            special_key = self._special_value_key(value)
            if special_key == "ref":
                return self._path_exists(context, value["ref"]) and (
                    self._lookup_path(context, value["ref"]) is not None
                )
            if special_key == "coalesce":
                return any(self._can_resolve(candidate, context) for candidate in value["coalesce"])
        return True

    @staticmethod
    def _lookup_path(context: Mapping[str, Any], path: str) -> Any:
        current: Any = context
        for segment in path.split("."):
            if isinstance(current, Mapping):
                current = current[segment]
            else:
                current = getattr(current, segment)
        return current

    @staticmethod
    def _path_exists(context: Mapping[str, Any], path: str) -> bool:
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

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else [value]

    def _interpolate_string(self, value: str, context: Mapping[str, Any]) -> Any:
        matches = list(PLACEHOLDER_RE.finditer(value))
        if not matches:
            return value
        if len(matches) == 1 and matches[0].span() == (0, len(value)):
            return self._lookup_path(context, matches[0].group(1))

        resolved = value
        for match in matches:
            resolved = resolved.replace(match.group(0), str(self._lookup_path(context, match.group(1))))
        return resolved

    def _dispatch_block(
        self,
        mapping: Mapping[str, str],
        block: Mapping[str, Any],
        args: tuple[Any, ...]
    ) -> None:
        handler_name = mapping.get(block["kind"])
        if handler_name is None:
            raise ValueError("Unsupported graph block kind '%s'." % block["kind"])
        getattr(self, handler_name)(*args)

    def _validate_operator_block(
        self,
        block: Mapping[str, Any],
        order_config: Mapping[str, Any]
    ) -> None:
        default_operator = BLOCK_DEFAULT_OPERATORS[block["kind"]]
        operator = self.operator_registry.resolve(block.get("operator", default_operator))
        if block["kind"] == "separate":
            operator.validate_params({"description": block["description"]})

    def _validate_chain_block(
        self,
        block: Mapping[str, Any],
        order_config: Mapping[str, Any]
    ) -> None:
        self._validate_chain(
            chain_name=block.get("name", block.get("prefix", block["kind"])),
            steps=block.get("steps", []),
            tag_order=self._chain_tag_order(block, order_config)
        )

    def _validate_step_block(
        self,
        block: Mapping[str, Any],
        order_config: Mapping[str, Any]
    ) -> None:
        self._validate_step(block)

    def _validate_chain(
        self,
        chain_name: str,
        steps: Sequence[Mapping[str, Any]],
        tag_order: Mapping[str, int]
    ) -> None:
        last_rank = -1
        for step in steps:
            self._validate_step(step)
            rank = self._operator_rank(self.operator_registry.resolve(step["operator"]).tags, tag_order)
            if rank is None:
                continue
            if rank < last_rank:
                raise ValueError(
                    "Chain '%s' violates configured operator ordering at '%s'." % (
                        chain_name,
                        step["operator"]
                    )
                )
            last_rank = rank

    def _operator_rank(
        self,
        operator_tags: Sequence[str],
        tag_order: Mapping[str, int]
    ) -> int | None:
        ranks = [
            tag_order[tag]
            for tag in operator_tags
            if tag in tag_order
        ]
        if not ranks:
            return None
        return min(ranks)

    def _chain_tag_order(
        self,
        block: Mapping[str, Any],
        order_config: Mapping[str, Any]
    ) -> Mapping[str, int]:
        profiles = order_config.get("profiles")
        if not profiles:
            return order_config.get("tags", order_config)
        profile_name = block.get("order_profile", order_config.get("default_profile"))
        if profile_name is None:
            return {}
        if profile_name not in profiles:
            raise ValueError("Unknown chain order profile '%s'." % profile_name)
        profile = profiles[profile_name]
        return profile.get("tags", profile)

    def _validate_step(self, step: Mapping[str, Any]) -> None:
        operator = self.operator_registry.resolve(step["operator"])
        operator.validate_params(step.get("params", {}))
