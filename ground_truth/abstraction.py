import re
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Iterator, Mapping, Sequence
import yaml
from typing import Any

from ground_truth.operators import OperatorRegistry

# Prefix marking a `bands_ref` that points into the `shared_bands` section.
SHARED_BANDS_PREFIX = "shared."

# `derives_from` value marking a level written from the symbolic graph itself.
GRAPH_SOURCE = "graph"

# Parameter roles. `magnitude` params also feed the chain intensity aggregate.
PARAM_ROLES = ("magnitude", "character", "ignore")

# How a parameter's value maps onto its bands and onto intensity.
PARAM_DIRECTIONS = ("ascending", "descending", "absolute")

# Distribution kinds and how to read their reachable support out of
# distributions.yaml. Mirrors DISTRIBUTION_HANDLER_NAMES in planner.py.
SUPPORT_HANDLER_NAMES = {
    "choice": "_choice_support",
    "grid": "_choice_support",
    "values": "_choice_support",
    "uniform": "_interval_support",
    "int_uniform": "_interval_support"
}

# Graph block kinds and the method that pulls parameter bearers out of them.
BLOCK_READER_NAMES = {
    "separate": "_read_operator_block",
    "mix": "_read_operator_block",
    "chain": "_read_steps_block",
    "send_return": "_read_send_return_block",
    "step": "_read_step_block"
}

# Default operators used when graph blocks omit operator fields. Mirrors
# BLOCK_DEFAULT_OPERATORS in planner.py.
BLOCK_DEFAULT_OPERATORS = {
    "separate": "separate_audio",
    "mix": "mix_stems"
}

# Routing operators that carry no audible character, so `must_mention_operators`
# does not require them to be named in prose.
STRUCTURAL_OPERATORS = ("separate_audio", "mix_stems")

# Alias decorations stripped when deriving the words prose is likely to use for
# an operator, e.g. `apply_highpass_filter` -> `highpass`.
ALIAS_PREFIXES = ("apply_", "introduce_")
ALIAS_SUFFIXES = ("_effect", "_filter", "_tool")

NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Band:
    upper: float
    suggested_terms: tuple[str, ...]


@dataclass(frozen=True)
class ParamSpec:
    operator: str
    param: str
    role: str
    direction: str = "ascending"
    unit: str | None = None
    weight: float = 1.0
    hint: str | None = None
    note: str | None = None
    sample_refs: tuple[str, ...] = field(default_factory=tuple)
    bands_ref: str | None = None
    bands: tuple[Band, ...] = field(default_factory=tuple)
    # Step label this spec is scoped to, for a parameter whose meaning depends on
    # which motif uses it. `None` is the default spec for the operator.
    label: str | None = None
    by_label: Mapping[str, "ParamSpec"] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        if self.label is None:
            return "%s.%s" % (self.operator, self.param)
        return "%s.%s@%s" % (self.operator, self.param, self.label)

    def for_label(self, label: str) -> "ParamSpec":
        return self.by_label.get(label, self)

    def band_index(self, value: float) -> tuple[int, bool]:
        """Return the band a value falls in, and whether it exceeded the top edge."""
        measured = self.measure(value)
        for index, band in enumerate(self.bands):
            if measured < band.upper:
                return index, False
        return len(self.bands) - 1, True

    def measure(self, value: float) -> float:
        return abs(float(value)) if self.direction == "absolute" else float(value)

    def intensity(self, band_index: int) -> float:
        # Position the value at the centre of its band's slot rather than on a
        # band edge. Edges would read a lone bottom-band parameter as zero
        # intensity, and collapse a single-band parameter onto an extreme; the
        # centre keeps both honest.
        position = (band_index + 0.5) / len(self.bands)
        return 1.0 - position if self.direction == "descending" else position


@dataclass(frozen=True)
class Support:
    """The set of values a parameter can actually be assigned."""

    points: tuple[float, ...] = field(default_factory=tuple)
    intervals: tuple[tuple[float, float], ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not self.points and not self.intervals

    def maximum(self) -> float:
        candidates = list(self.points)
        candidates.extend(high for _, high in self.intervals)
        return max(candidates)

    def intersects(self, low: float, high: float) -> bool:
        if any(low <= point < high for point in self.points):
            return True
        return any(
            interval_low < high and low < interval_high
            for interval_low, interval_high in self.intervals
        )

    def absolute(self) -> "Support":
        points = tuple(abs(point) for point in self.points)
        intervals: list[tuple[float, float]] = []
        for low, high in self.intervals:
            if low <= 0.0 <= high:
                intervals.append((0.0, max(abs(low), abs(high))))
            else:
                intervals.append((min(abs(low), abs(high)), max(abs(low), abs(high))))
        return Support(points=points, intervals=tuple(intervals))

    def merge(self, other: "Support") -> "Support":
        return Support(
            points=self.points + other.points,
            intervals=self.intervals + other.intervals
        )


@dataclass(frozen=True)
class ParamDescriptor:
    """One parameter of one chain step, resolved to its descriptor band."""

    label: str
    operator: str
    param: str
    value: Any
    role: str
    unit: str | None
    band_terms: tuple[str, ...]
    hint: str | None
    intensity: float | None
    weight: float
    out_of_band: bool


@dataclass(frozen=True)
class ChainMagnitude:
    value: float | None
    band: str | None
    suggested_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": None if self.value is None else round(self.value, 4),
            "band": self.band,
            "suggested_terms": list(self.suggested_terms)
        }


@dataclass(frozen=True)
class ChainProfile:
    """Everything derived from a graph spec that prompt generation needs."""

    descriptors: tuple[ParamDescriptor, ...]
    unbanded: tuple[ParamDescriptor, ...]
    operators: tuple[str, ...]
    tags: tuple[str, ...]
    magnitude: ChainMagnitude
    # Chain-level vocabulary from the context overlays. This is deliberately not
    # merged into the per-parameter bands: the overlays describe the character of
    # the chain as a whole, and a term like "telephone-like" is not a descriptor
    # for reverb damping.
    overlay_terms: tuple[str, ...] = field(default_factory=tuple)
    operator_tags: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def param_values(self) -> tuple[tuple[str, Any], ...]:
        return tuple(
            ("%s.%s" % (item.operator, item.param), item.value)
            for item in (*self.descriptors, *self.unbanded)
        )

    def band_assignments(self) -> dict[str, str]:
        return {
            "%s.%s" % (item.label, item.param): item.band_terms[0]
            for item in self.descriptors
            if item.band_terms
        }


class BandLexicon:
    """Parameter descriptor bands, grounded in the sampling distributions."""

    def __init__(
        self,
        param_bands: Mapping[str, Mapping[str, ParamSpec]],
        block_bands: Mapping[str, Mapping[str, ParamSpec]],
        overlays: Sequence[Mapping[str, Any]],
        magnitude_bands: Sequence[Band],
        shared_bands: Mapping[str, tuple[Band, ...]],
        version: int = 1
    ):
        self.version = version
        self._param_bands = {
            operator: dict(params) for operator, params in param_bands.items()
        }
        self._block_bands = {
            kind: dict(params) for kind, params in block_bands.items()
        }
        self._overlays = [dict(overlay) for overlay in overlays]
        self._magnitude_bands = tuple(magnitude_bands)
        self._shared_bands = dict(shared_bands)

    @classmethod
    def from_config(cls, config_path: Path | str) -> "BandLexicon":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}

        shared_bands = {
            name: cls._parse_bands(definition, "shared_bands.%s" % name)
            for name, definition in (loaded.get("shared_bands") or {}).items()
        }
        param_bands = {
            operator: {
                param: cls._parse_param_spec(operator, param, definition, shared_bands)
                for param, definition in (params or {}).items()
            }
            for operator, params in (loaded.get("param_bands") or {}).items()
        }
        block_bands = {
            kind: {
                param: cls._parse_param_spec(kind, param, definition, shared_bands)
                for param, definition in (params or {}).items()
            }
            for kind, params in (loaded.get("block_bands") or {}).items()
        }
        magnitude_bands = cls._parse_bands(
            (loaded.get("chain_magnitude") or {}).get("bands", []),
            "chain_magnitude.bands"
        )
        if not magnitude_bands:
            raise ValueError("Band config '%s' does not define chain_magnitude bands." % path)

        return cls(
            param_bands=param_bands,
            block_bands=block_bands,
            overlays=loaded.get("context_overlays") or [],
            magnitude_bands=magnitude_bands,
            shared_bands=shared_bands,
            version=int(loaded.get("version", 1))
        )

    def validate_against_operators(self, registry: OperatorRegistry) -> None:
        """Reject bands for operators or parameters that do not exist."""
        for operator, params in self._param_bands.items():
            spec = registry.resolve(operator)
            unknown = sorted(set(params) - set(spec.params))
            if unknown:
                raise ValueError(
                    "Band config declares parameter(s) %s that operator '%s' does not accept." % (
                        ", ".join(unknown),
                        operator
                    )
                )

    def validate_against_distributions(
        self,
        distributions: Mapping[str, Any],
        literals: Mapping[str, Support] | None = None
    ) -> None:
        """Reject band sets that do not line up with the reachable supports.

        Two failure modes are caught: a band set that does not cover every value
        a parameter can be assigned, and a band that no reachable value can ever
        hit, which would put dead vocabulary in front of the writer.

        `literals` carries the constant parameter values written directly into
        motifs.yaml and recipes.yaml, which never pass through a distribution;
        without them a band set can miss part of its real range.
        """
        shared_support: dict[str, Support] = {}
        for spec in self.iter_specs():
            if spec.role == "ignore" or not spec.bands:
                continue
            support = self._resolve_support(spec, distributions)
            literal = (literals or {}).get(spec.qualified_name)
            if literal is not None:
                support = support.merge(
                    literal.absolute() if spec.direction == "absolute" else literal
                )
            if support.is_empty():
                continue
            self._check_coverage(spec, support)
            if spec.bands_ref is None:
                self._check_reachable(spec, support, spec.bands)
            else:
                merged = shared_support.get(spec.bands_ref, Support())
                shared_support[spec.bands_ref] = merged.merge(support)

        for bands_ref, support in shared_support.items():
            bands = self._lookup_shared_bands(bands_ref)
            self._check_reachable(bands_ref, support, bands)

    def iter_specs(self) -> Iterator[ParamSpec]:
        for params in (*self._param_bands.values(), *self._block_bands.values()):
            for spec in params.values():
                yield spec
                yield from spec.by_label.values()

    def param_spec(self, operator: str, param: str) -> ParamSpec | None:
        return self._param_bands.get(operator, {}).get(param)

    def block_spec(self, kind: str, param: str) -> ParamSpec | None:
        return self._block_bands.get(kind, {}).get(param)

    def overlay_terms(self, tags: Sequence[str]) -> tuple[str, ...]:
        present = set(tags)
        terms: list[str] = []
        for overlay in self._overlays:
            if present.intersection(overlay.get("when_chain_has_tags", [])):
                terms.extend(overlay.get("add_terms", []))
        return tuple(dict.fromkeys(terms))

    def magnitude_band(self, value: float) -> Band:
        for band in self._magnitude_bands:
            if value < band.upper:
                return band
        return self._magnitude_bands[-1]

    @staticmethod
    def _parse_bands(definitions: Any, context: str) -> tuple[Band, ...]:
        bands: list[Band] = []
        previous_upper: float | None = None
        for definition in definitions or []:
            if "upper" not in definition:
                raise ValueError("Band in '%s' is missing an 'upper' edge." % context)
            upper = float(definition["upper"])
            if previous_upper is not None and upper <= previous_upper:
                raise ValueError(
                    "Band edges in '%s' must strictly increase; %s follows %s." % (
                        context,
                        upper,
                        previous_upper
                    )
                )
            terms = tuple(definition.get("suggested_terms", []))
            if not terms:
                raise ValueError("Band in '%s' declares no suggested_terms." % context)
            bands.append(Band(upper=upper, suggested_terms=terms))
            previous_upper = upper
        return tuple(bands)

    @classmethod
    def _parse_param_spec(
        cls,
        owner: str,
        param: str,
        definition: Mapping[str, Any],
        shared_bands: Mapping[str, tuple[Band, ...]],
        label: str | None = None
    ) -> ParamSpec:
        context = "%s.%s" % (owner, param) if label is None else "%s.%s@%s" % (
            owner,
            param,
            label
        )
        role = definition.get("role", "character")
        if role not in PARAM_ROLES:
            raise ValueError(
                "Parameter '%s' declares unsupported role '%s'." % (context, role)
            )
        direction = definition.get("direction", "ascending")
        if direction not in PARAM_DIRECTIONS:
            raise ValueError(
                "Parameter '%s' declares unsupported direction '%s'." % (context, direction)
            )

        bands_ref = definition.get("bands_ref")
        if bands_ref is not None and "bands" in definition:
            raise ValueError(
                "Parameter '%s' declares both 'bands' and 'bands_ref'." % context
            )
        if bands_ref is not None:
            if not bands_ref.startswith(SHARED_BANDS_PREFIX):
                raise ValueError(
                    "Parameter '%s' has bands_ref '%s' outside '%s'." % (
                        context,
                        bands_ref,
                        SHARED_BANDS_PREFIX
                    )
                )
            name = bands_ref[len(SHARED_BANDS_PREFIX):]
            if name not in shared_bands:
                raise ValueError(
                    "Parameter '%s' references unknown shared band set '%s'." % (context, name)
                )
            bands = shared_bands[name]
        else:
            bands = cls._parse_bands(definition.get("bands", []), context)

        if role != "ignore" and not bands:
            raise ValueError("Parameter '%s' has role '%s' but no bands." % (context, role))

        # A label override inherits everything it does not restate, so it only
        # has to declare what differs for that motif.
        by_label: dict[str, ParamSpec] = {}
        for override_label, override in (definition.get("by_label") or {}).items():
            if label is not None:
                raise ValueError("Parameter '%s' nests a by_label override." % context)
            merged = {
                key: value for key, value in definition.items() if key != "by_label"
            }
            merged.update(override)
            if "bands" in override:
                merged.pop("bands_ref", None)
            if "bands_ref" in override:
                merged.pop("bands", None)
            by_label[override_label] = cls._parse_param_spec(
                owner=owner,
                param=param,
                definition=merged,
                shared_bands=shared_bands,
                label=override_label
            )

        return ParamSpec(
            operator=owner,
            param=param,
            role=role,
            direction=direction,
            unit=definition.get("unit"),
            weight=float(definition.get("weight", 1.0)),
            hint=definition.get("hint"),
            note=definition.get("note"),
            sample_refs=tuple(definition.get("sample_refs", []) or []),
            bands_ref=bands_ref,
            bands=bands,
            label=label,
            by_label=by_label
        )

    def _lookup_shared_bands(self, bands_ref: str) -> tuple[Band, ...]:
        return self._shared_bands[bands_ref[len(SHARED_BANDS_PREFIX):]]

    def _resolve_support(
        self,
        spec: ParamSpec,
        distributions: Mapping[str, Any]
    ) -> Support:
        support = Support()
        for ref in spec.sample_refs:
            support = support.merge(self._distribution_support(ref, distributions))
        return support.absolute() if spec.direction == "absolute" else support

    @classmethod
    def _distribution_support(
        cls,
        ref: str,
        distributions: Mapping[str, Any]
    ) -> Support:
        node: Any = distributions
        for part in ref.split("."):
            if not isinstance(node, Mapping) or part not in node:
                raise ValueError("Unknown distribution '%s' in band config." % ref)
            node = node[part]
        if not isinstance(node, Mapping):
            raise ValueError("Distribution '%s' is not a distribution node." % ref)

        distribution_type = node.get("type", "choice")
        handler_name = SUPPORT_HANDLER_NAMES.get(distribution_type)
        if handler_name is None:
            raise ValueError(
                "Unsupported distribution type '%s' for '%s'." % (distribution_type, ref)
            )
        return getattr(cls, handler_name)(node)

    @staticmethod
    def _choice_support(spec: Mapping[str, Any]) -> Support:
        points: list[float] = []
        for item in spec.get("values", []):
            value = item["value"] if isinstance(item, Mapping) and "value" in item else item
            if isinstance(value, (int, float)):
                points.append(float(value))
        return Support(points=tuple(points))

    @staticmethod
    def _interval_support(spec: Mapping[str, Any]) -> Support:
        low = float(spec["low"])
        high = float(spec["high"])
        # The declared `samples` are the auditable landmarks band edges are cut
        # against, so carry them alongside the continuous range.
        points = tuple(
            float(sample)
            for sample in spec.get("samples", [])
            if isinstance(sample, (int, float))
        )
        return Support(points=points, intervals=((low, high),))

    @staticmethod
    def _check_coverage(spec: ParamSpec, support: Support) -> None:
        top_edge = spec.bands[-1].upper
        maximum = support.maximum()
        if maximum >= top_edge:
            raise ValueError(
                "Bands for '%s' stop at %s but its sampled support reaches %s." % (
                    spec.qualified_name,
                    top_edge,
                    maximum
                )
            )

    @staticmethod
    def _check_reachable(
        owner: ParamSpec | str,
        support: Support,
        bands: Sequence[Band]
    ) -> None:
        name = owner if isinstance(owner, str) else owner.qualified_name
        lower = float("-inf")
        for band in bands:
            if not support.intersects(lower, band.upper):
                raise ValueError(
                    "Band [%s, %s) of '%s' is unreachable from the sampled support." % (
                        lower,
                        band.upper,
                        name
                    )
                )
            lower = band.upper


class ChainReader:
    """Turns a symbolic graph spec into descriptor bands and a chain intensity."""

    def __init__(self, lexicon: BandLexicon, registry: OperatorRegistry):
        self.lexicon = lexicon
        self.registry = registry

    def profile(self, graph_spec: Sequence[Mapping[str, Any]]) -> ChainProfile:
        descriptors: list[ParamDescriptor] = []
        unbanded: list[ParamDescriptor] = []
        operators: list[str] = []
        operator_tags: dict[str, tuple[str, ...]] = {}

        bearers = list(self._read_blocks(graph_spec))
        for label, owner, is_block, params in bearers:
            if not is_block:
                operators.append(owner)
                if self.registry.has(owner):
                    operator_tags[owner] = tuple(self.registry.resolve(owner).tags)
            for param, value in sorted(params.items()):
                spec = (
                    self.lexicon.block_spec(owner, param)
                    if is_block
                    else self.lexicon.param_spec(owner, param)
                )
                if spec is not None and spec.role == "ignore":
                    continue
                if spec is None or not isinstance(value, (int, float)):
                    unbanded.append(
                        self._unbanded_descriptor(label, owner, param, value)
                    )
                    continue
                descriptors.append(self._banded_descriptor(label, spec, value))

        # Routing operators contribute no character, so their tags are dropped
        # rather than offered to the writer as chain character.
        chain_tags = tuple(
            dict.fromkeys(
                tag
                for operator, owner_tags in operator_tags.items()
                if operator not in STRUCTURAL_OPERATORS
                for tag in owner_tags
            )
        )
        return ChainProfile(
            descriptors=tuple(descriptors),
            unbanded=tuple(unbanded),
            operators=tuple(dict.fromkeys(operators)),
            tags=chain_tags,
            magnitude=self._magnitude(descriptors),
            overlay_terms=self.lexicon.overlay_terms(chain_tags),
            operator_tags=operator_tags
        )

    def _banded_descriptor(
        self,
        label: str,
        spec: ParamSpec,
        value: float
    ) -> ParamDescriptor:
        spec = spec.for_label(label)
        index, out_of_band = spec.band_index(value)
        terms = spec.bands[index].suggested_terms
        return ParamDescriptor(
            label=label,
            operator=spec.operator,
            param=spec.param,
            value=value,
            role=spec.role,
            unit=spec.unit,
            band_terms=terms,
            hint=spec.hint,
            intensity=spec.intensity(index) if spec.role == "magnitude" else None,
            weight=spec.weight,
            out_of_band=out_of_band
        )

    @staticmethod
    def _unbanded_descriptor(
        label: str,
        owner: str,
        param: str,
        value: Any
    ) -> ParamDescriptor:
        return ParamDescriptor(
            label=label,
            operator=owner,
            param=param,
            value=value,
            role="character",
            unit=None,
            band_terms=(),
            hint=None,
            intensity=None,
            weight=0.0,
            out_of_band=False
        )

    def _magnitude(self, descriptors: Sequence[ParamDescriptor]) -> ChainMagnitude:
        weighted = [
            (descriptor.weight, descriptor.intensity)
            for descriptor in descriptors
            if descriptor.intensity is not None and descriptor.weight > 0.0
        ]
        if not weighted:
            return ChainMagnitude(value=None, band=None, suggested_terms=())
        total_weight = sum(weight for weight, _ in weighted)
        value = sum(weight * intensity for weight, intensity in weighted) / total_weight
        band = self.lexicon.magnitude_band(value)
        return ChainMagnitude(
            value=value,
            band=band.suggested_terms[0],
            suggested_terms=band.suggested_terms
        )

    def _read_blocks(
        self,
        graph_spec: Sequence[Mapping[str, Any]]
    ) -> Iterator[tuple[str, str, bool, Mapping[str, Any]]]:
        for block in graph_spec:
            kind = block.get("kind")
            handler_name = BLOCK_READER_NAMES.get(kind)
            if handler_name is None:
                raise ValueError("Unsupported graph block kind '%s'." % kind)
            yield from getattr(self, handler_name)(block)

    @staticmethod
    def _read_operator_block(
        block: Mapping[str, Any]
    ) -> Iterator[tuple[str, str, bool, Mapping[str, Any]]]:
        operator = block.get("operator", BLOCK_DEFAULT_OPERATORS.get(block["kind"]))
        yield block.get("name", operator), operator, False, block.get("params", {})

    @classmethod
    def _read_steps_block(
        cls,
        block: Mapping[str, Any]
    ) -> Iterator[tuple[str, str, bool, Mapping[str, Any]]]:
        for step in block.get("steps", []):
            yield from cls._read_step_block(step)

    @staticmethod
    def _read_step_block(
        block: Mapping[str, Any]
    ) -> Iterator[tuple[str, str, bool, Mapping[str, Any]]]:
        operator = block["operator"]
        yield block.get("name", operator), operator, False, block.get("params", {})

    @classmethod
    def _read_send_return_block(
        cls,
        block: Mapping[str, Any]
    ) -> Iterator[tuple[str, str, bool, Mapping[str, Any]]]:
        name = block.get("name", "send_return")
        levels = {
            param: block[param]
            for param in ("dry_level", "send_level", "return_level")
            if param in block
        }
        if levels:
            yield name, "send_return", True, levels
        yield from cls._read_steps_block(block)


@dataclass(frozen=True)
class AbstractionLevel:
    id: int
    name: str
    derives_from: int | str
    constraints: Mapping[str, Any] = field(default_factory=dict)
    # The voice this level is written in. `None` falls back to the ladder's.
    system_prompt: str | None = None
    rules: tuple[str, ...] = field(default_factory=tuple)
    exemplars: tuple[str, ...] = field(default_factory=tuple)
    hard_checks: Mapping[str, Any] = field(default_factory=dict)
    rubric: tuple[str, ...] = field(default_factory=tuple)
    max_tokens: int = 256

    @property
    def from_graph(self) -> bool:
        return self.derives_from == GRAPH_SOURCE


class AbstractionLadder:
    """The declared abstraction levels and the checks that police them."""

    def __init__(
        self,
        levels: Sequence[AbstractionLevel],
        system_prompt: str,
        version: int,
        output_contract: str = ""
    ):
        self.system_prompt = system_prompt
        self.output_contract = output_contract
        self.version = version
        self._levels = {level.id: level for level in levels}
        if len(self._levels) != len(levels):
            raise ValueError("Abstraction levels declare duplicate ids.")
        self._order = self._resolve_order(levels)

    @classmethod
    def from_config(cls, config_path: Path | str) -> "AbstractionLadder":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        definitions = loaded.get("levels", [])
        if not definitions:
            raise ValueError("Abstraction config '%s' does not define any levels." % path)

        levels = []
        for definition in definitions:
            checks = definition.get("checks", {}) or {}
            levels.append(
                AbstractionLevel(
                    id=int(definition["id"]),
                    name=definition["name"],
                    derives_from=definition.get("derives_from", GRAPH_SOURCE),
                    constraints=dict(definition.get("constraints", {})),
                    system_prompt=definition.get("system_prompt"),
                    rules=tuple(definition.get("rules", [])),
                    exemplars=tuple(definition.get("exemplars", [])),
                    hard_checks=dict(checks.get("hard", {})),
                    rubric=tuple(checks.get("rubric", [])),
                    max_tokens=int(definition.get("max_tokens", 256))
                )
            )
        return cls(
            levels=levels,
            system_prompt=loaded.get("system_prompt", "You are a helpful, concise assistant."),
            version=int(loaded.get("version", 1)),
            output_contract=loaded.get("output_contract", "")
        )

    def levels(self) -> tuple[AbstractionLevel, ...]:
        return self._order

    def system_prompt_for(self, level: AbstractionLevel) -> str:
        """The voice a level is written in, plus the shared output contract."""
        parts = (level.system_prompt or self.system_prompt, self.output_contract)
        return " ".join(part.strip() for part in parts if part and part.strip())

    def select(self, level_ids: Sequence[int]) -> tuple[AbstractionLevel, ...]:
        """Return the requested levels plus every level they derive from."""
        required: set[int] = set()
        for level_id in level_ids:
            if level_id not in self._levels:
                raise ValueError("Unknown abstraction level '%s'." % level_id)
            cursor: int | str = level_id
            while cursor != GRAPH_SOURCE:
                required.add(int(cursor))
                cursor = self._levels[int(cursor)].derives_from
        return tuple(level for level in self._order if level.id in required)

    def check(
        self,
        level: AbstractionLevel,
        text: str,
        profile: ChainProfile
    ) -> tuple[str, ...]:
        """Run the level's mechanical checks and return any violations."""
        violations: list[str] = []
        checks = level.hard_checks

        for pattern in checks.get("forbid_patterns", []):
            match = re.search(pattern, text)
            if match:
                violations.append(
                    "matched forbidden pattern %s (%r)" % (pattern, match.group(0))
                )

        max_words = checks.get("max_words")
        if max_words is not None:
            word_count = len(text.split())
            if word_count > int(max_words):
                violations.append("used %d words, limit is %s" % (word_count, max_words))

        if checks.get("must_mention_operators"):
            missing = self._missing_operators(
                text=text,
                operators=profile.operators,
                operator_tags=profile.operator_tags,
                exempt_tags=checks.get("mention_exempt_tags", [])
            )
            if missing:
                violations.append("did not name operator(s) %s" % ", ".join(missing))

        if checks.get("must_contain_all_param_values"):
            missing_values = self._missing_values(text, profile)
            if missing_values:
                violations.append("omitted value(s) %s" % ", ".join(missing_values))

        return tuple(violations)

    def _missing_operators(
        self,
        text: str,
        operators: Sequence[str],
        operator_tags: Mapping[str, tuple[str, ...]],
        exempt_tags: Sequence[str]
    ) -> list[str]:
        normalized = NON_ALNUM_RE.sub("", text.lower())
        exempt = set(exempt_tags)
        missing: list[str] = []
        for operator in operators:
            if operator in STRUCTURAL_OPERATORS:
                continue
            if exempt.intersection(operator_tags.get(operator, ())):
                continue
            if not any(token in normalized for token in self._operator_tokens(operator)):
                missing.append(operator)
        return missing

    @staticmethod
    def _operator_tokens(operator: str) -> tuple[str, ...]:
        stripped = operator
        for prefix in ALIAS_PREFIXES:
            stripped = stripped.removeprefix(prefix)
        for suffix in ALIAS_SUFFIXES:
            stripped = stripped.removesuffix(suffix)
        candidates = {stripped, stripped.split("_")[-1]}
        return tuple(
            NON_ALNUM_RE.sub("", candidate.lower())
            for candidate in candidates
            if candidate
        )

    @classmethod
    def _missing_values(cls, text: str, profile: ChainProfile) -> list[str]:
        normalized = NON_ALNUM_RE.sub("", text.lower())
        missing: list[str] = []
        for name, value in profile.param_values():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            renderings = cls._value_renderings(float(value))
            if not any(rendering in normalized for rendering in renderings):
                missing.append("%s=%s" % (name, value))
        return missing

    @staticmethod
    def _value_renderings(value: float) -> tuple[str, ...]:
        magnitude = abs(value)
        plain = "%g" % magnitude
        renderings = {plain, plain.lstrip("0") or plain}
        if magnitude >= 1000.0:
            renderings.add("%gk" % (magnitude / 1000.0))
        if magnitude.is_integer():
            renderings.add("%d" % int(magnitude))
        return tuple(
            NON_ALNUM_RE.sub("", rendering.lower())
            for rendering in renderings
            if rendering
        )

    @staticmethod
    def _resolve_order(levels: Sequence[AbstractionLevel]) -> tuple[AbstractionLevel, ...]:
        by_id = {level.id: level for level in levels}
        ordered: list[AbstractionLevel] = []
        resolved: set[int] = set()
        pending = list(levels)
        while pending:
            progressed = False
            for level in list(pending):
                source = level.derives_from
                if source != GRAPH_SOURCE and int(source) not in by_id:
                    raise ValueError(
                        "Level %s derives from unknown level '%s'." % (level.id, source)
                    )
                if source == GRAPH_SOURCE or int(source) in resolved:
                    ordered.append(level)
                    resolved.add(level.id)
                    pending.remove(level)
                    progressed = True
            if not progressed:
                raise ValueError(
                    "Abstraction levels %s form a derives_from cycle." % (
                        ", ".join(str(level.id) for level in pending)
                    )
                )
        return tuple(ordered)


def literal_param_support(*sections: Any) -> dict[str, Support]:
    """Collect constant parameter values written directly into config sections.

    Motifs and recipes may pin a parameter to a literal instead of sampling it,
    e.g. `send_level: 1.0` on a send_return block. Those values are reachable but
    appear in no distribution, so band coverage has to account for them too.
    """
    collected: dict[str, list[float]] = {}

    def record(owner: str, params: Mapping[str, Any]) -> None:
        for param, value in params.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            collected.setdefault("%s.%s" % (owner, param), []).append(float(value))

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            if "operator" in node and isinstance(node.get("params"), Mapping):
                record(str(node["operator"]), node["params"])
            if node.get("kind") == "send_return":
                record(
                    "send_return",
                    {
                        param: node[param]
                        for param in ("dry_level", "send_level", "return_level")
                        if param in node
                    }
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for value in node:
                walk(value)

    for section in sections:
        walk(section)
    return {
        name: Support(points=tuple(values)) for name, values in collected.items()
    }


def _load_config_section(path: Path, root_key: str) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return loaded.get(root_key, {})


def load_abstraction_config(
    config_dir: Path | str,
    registry: OperatorRegistry,
    distributions: Mapping[str, Any] | None = None,
    literals: Mapping[str, Support] | None = None,
    levels_config: Path | str | None = None,
    bands_config: Path | str | None = None
) -> tuple[AbstractionLadder, BandLexicon]:
    """Load the abstraction ladder and band lexicon, validating the grounding.

    `distributions` and `literals` are read out of `config_dir` when not given.
    Both are needed for validation to be meaningful, so passing only one of them
    is a good way to get a spurious failure; leaving both alone is the norm.
    """
    config_root = Path(config_dir)
    if distributions is None:
        distributions = _load_config_section(
            config_root / "distributions.yaml", "distributions"
        )
    if literals is None:
        literals = literal_param_support(
            _load_config_section(config_root / "motifs.yaml", "motifs"),
            _load_config_section(config_root / "recipes.yaml", "recipes")
        )

    ladder = AbstractionLadder.from_config(
        levels_config or config_root / "abstraction_levels.yaml"
    )
    lexicon = BandLexicon.from_config(
        bands_config or config_root / "param_bands.yaml"
    )
    lexicon.validate_against_operators(registry)
    lexicon.validate_against_distributions(distributions, literals)
    return ladder, lexicon
