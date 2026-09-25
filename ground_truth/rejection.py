"""Rejection filtering for generated ground-truth rows.

Stage 5 of the ground-truth stack. Everything here is model-free: config
loading, the trigger context, deterministic trigger evaluation, the reachability
derivation, prompt assembly and response parsing. The Bedrock call itself lives
in scripts/filter_ground_truth_prompts.py, beside the CLI that decides how many
of them to make.

The split that matters: `trigger` predicates are evaluated here, exactly, and
decide whether a judge call happens at all. The judge only adjudicates rules
that already fired, and is handed their statements, their sources and the actual
parameter values, so it never has to do arithmetic.
"""

import re
import json
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Mapping, Iterator, Sequence
from ground_truth.operators import OperatorRegistry
from ground_truth.predicates import evaluate_condition, iter_condition_paths
from ground_truth.abstraction import GRAPH_SOURCE, BLOCK_DEFAULT_OPERATORS, Support, BandLexicon, ChainReader, AbstractionLadder, literal_param_support
from ground_truth.summarization import ModelSettings
import yaml
from typing import Any

ROOT_KEY = "rejection"

CONFIG_FILE_NAME = "rejection.yaml"

# The bpm the tempo_sync motif falls back to when a clip reports no tempo. This
# mirrors the `coalesce` literal in motifs.yaml; inverting delay_seconds back to
# beats is only correct while the two agree.
DEFAULT_TEMPO_BPM = 120.0

# Context roots a rule's `trigger`/`detect` paths may address, and how deep the
# fixed part of the path runs before data-dependent keys begin. A path outside
# these is a config error, not a dormant rule.
CONTEXT_ROOTS = ("params", "plan", "profile", "caption", "analysis", "derived")

# Aggregations under `params`. `max` and `min` collapse repeated occurrences of
# the same operator in one chain, which the condition DSL cannot do itself; the
# block scopes narrow to one kind of graph block first.
PARAM_SCOPES = ("max", "min", "label", "send", "chain")

PARAM_AGGREGATES = ("max", "min")

# How a comparison key bounds the region a trigger selects, as (low, high) with
# None meaning unbounded. Used only by the reachability derivation.
COMPARISON_REGIONS = {
    "gte": lambda value: (value, None),
    "gt": lambda value: (value, None),
    "lte": lambda value: (None, value),
    "lt": lambda value: (None, value),
    "eq": lambda value: (value, value),
    "neq": lambda value: (None, None)
}

# Reachability verdicts, worst-first for reporting. The distinction between the
# middle two is the point of the exercise: a narrow distribution re-arms a rule
# when someone widens it, a missing operator or control never does.
REACH_LIVE = "live"
REACH_VACUOUS = "vacuous"
REACH_DORMANT_DISTRIBUTION = "dormant_distribution"
REACH_DORMANT_NO_PARAM = "dormant_no_param"
REACH_UNREACHABLE_OPERATOR = "unreachable_operator"
REACH_DATA_DEPENDENT = "data_dependent"

# Graph-block fields whose values name something inside the edit graph:
# step labels, stem handles, routing endpoints. Leaking any of them into an
# instruction is leaking the implementation.
GRAPH_IDENTIFIER_FIELDS = ("name", "prefix", "source", "output", "stem", "residual", "description")

# How a rewrite can fail to describe the level below it. Not mutually
# exclusive across a whole rewrite, so these are reported as a list.
#
#   wrong_interpretation  the element IS in the source but is described wrongly
#                         -- wrong direction, wrong magnitude band, wrong stem,
#                         wrong processor. Information present, but wrong.
#   hallucination         processing, a parameter or a target that is not in
#                         the source at all. Added, not distorted.
#   omission              something the level's contract obliged it to carry is
#                         absent. NOT "less detailed": discarding what the
#                         contract permits is the contract working.
CONSISTENCY_FAILURES = ("wrong_interpretation", "hallucination", "omission")

VERDICT_LEAKED = "leaked"
VERDICT_CLEAN = "clean"

# Verdict dispositions.
DISPOSITION_ACCEPT = "accept"
DISPOSITION_REJECT = "reject"
DISPOSITION_REVIEW = "review"

UNCLEAR_ACTIONS = ("keep", "reject", "review")

# Which manifest field supplies the A-side of the comparison.
REFERENCE_MODES = ("baseline_then_source", "baseline_only", "source_only")

# First balanced `{...}` run in a model response. Models wrap JSON in prose and
# code fences however they like, so the parse is lenient by design and the
# schema check downstream is what actually rejects a bad response.
JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

WORD_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Rule:
    """One sourced rule with a deterministic trigger."""

    id: str
    statement: str
    source: str
    kind: str
    trigger: Mapping[str, Any] | None
    suggests: str | None = None
    applies_to: Mapping[str, Any] = field(default_factory=dict)
    signature: str | None = None
    periods: tuple[str, ...] = field(default_factory=tuple)
    genres: tuple[str, ...] = field(default_factory=tuple)

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(iter_condition_paths(self.trigger))


@dataclass(frozen=True)
class Criterion:
    """One judged criterion and the rules it adjudicates."""

    id: str
    produces: tuple[str, ...]
    per: str
    output: str
    default_verdict: str
    system_prompt: str
    rules: tuple[str, ...]
    exemplars: tuple[Mapping[str, Any], ...]
    max_tokens: int
    values: tuple[str, ...] = field(default_factory=tuple)
    requires: tuple[str, ...] = field(default_factory=tuple)
    rule_set: tuple[Rule, ...] = field(default_factory=tuple)

    def rule(self, rule_id: str) -> Rule | None:
        for item in self.rule_set:
            if item.id == rule_id:
                return item
        return None


@dataclass(frozen=True)
class MertSimilarityConfig:
    """How to measure the distance between the original and the edited audio."""

    model: str = "MERT-v1-95M"
    # "all" means mean over the layer stack; an int takes that layer alone. 12
    # reproduces an unmodified fadtk.MERTModel(), which is the setting under
    # which these numbers are comparable to the lab's existing FAD/KAD run.
    layer: str = "all"
    device: str | None = None
    render_manifest: Path | None = None
    cache_dir: Path = Path("derived/ground_truth/mert_cache")
    similarity_path: Path = Path("derived/ground_truth/mert_similarity.jsonl")
    reference: str = "baseline_then_source"
    # Null keeps the criterion descriptive: the value is recorded and the flag's
    # verdict stays None, so nothing is rejected on a threshold nobody has
    # chosen from data yet.
    similarity_threshold: float | None = None

    @property
    def numeric_layer(self) -> int:
        """The layer fadtk should select, for the single-layer case."""
        return 12 if self.pools_all_layers else int(self.layer)

    @property
    def pools_all_layers(self) -> bool:
        return str(self.layer).lower() == "all"

    @property
    def model_key(self) -> str:
        """Model identity for the cache key, so a layer change invalidates it."""
        return "%s-layer_%s" % (self.model, self.layer)

    @classmethod
    def from_mapping(cls, section: Mapping[str, Any], context: str) -> "MertSimilarityConfig":
        layer = section.get("layer", "all")
        if str(layer).lower() != "all":
            try:
                index = int(layer)
            except (TypeError, ValueError):
                raise ValueError("Rejection config '%s' sets mert_similarity.layer to %r; expected 'all' or an integer 1-12." % (context, layer)) from None
            if not 1 <= index <= 12:
                raise ValueError("Rejection config '%s' sets mert_similarity.layer to %d; MERT-v1-95M has layers 1-12." % (context, index))
        reference = section.get("reference", "baseline_then_source")
        if reference not in REFERENCE_MODES:
            raise ValueError("Rejection config '%s' sets mert_similarity.reference to '%s'; expected one of %s." % (context, reference, ", ".join(REFERENCE_MODES)))
        threshold = section.get("similarity_threshold")
        if threshold is not None:
            threshold = float(threshold)
            if not -1.0 <= threshold <= 1.0:
                raise ValueError("Rejection config '%s' sets mert_similarity.similarity_threshold to %s; a cosine similarity lies in [-1, 1]." % (context, threshold))
        return cls(
            model=str(section.get("model", "MERT-v1-95M")),
            layer=str(layer),
            device=(str(section["device"]).strip() or None) if section.get("device") else None,
            render_manifest=_optional_path(section, "render_manifest"),
            cache_dir=Path(str(section.get("cache_dir", "derived/ground_truth/mert_cache"))).expanduser(),
            similarity_path=Path(str(section.get("similarity_path", "derived/ground_truth/mert_similarity.jsonl"))).expanduser(),
            reference=str(reference),
            similarity_threshold=threshold
        )


@dataclass(frozen=True)
class RejectionConfig:
    """Everything the rejection stage runs by."""

    model: ModelSettings
    prompts_path: Path
    analysis_path: Path | None
    output_path: Path
    review_path: Path | None
    failures_path: Path | None
    full_manifest_path: Path | None
    report_path: Path | None
    captions_path: Path | None
    mert: MertSimilarityConfig
    max_attempts: int
    max_workers: int
    criteria: tuple[Criterion, ...]
    criteria_filter: tuple[str, ...]
    subsample_num: int
    seed: int
    reject_verdicts: Mapping[str, tuple[str, ...]]
    unclear_actions: Mapping[str, str]
    on_judge_error: str
    output_contract: str
    constraint_glosses: Mapping[str, Mapping[str, str]]
    caption_cues: Mapping[str, tuple[str, ...]]
    leak_syntax_patterns: tuple[str, ...]
    leak_allow: tuple[str, ...]
    # Which consistency failure types actually reject, as opposed to being
    # recorded. Kept separate from `reject_verdicts` because the verdict stays
    # an honest "did it describe the source correctly", while whether a given
    # kind of wrongness is disqualifying is a policy call.
    reject_failures: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    version: int = 1
    # Inputs a criterion may declare in `requires` that this run cannot supply,
    # so criteria depending on them are dropped rather than run to a null
    # verdict. Dropping is not the same as letting them report `no_caption`:
    # that verdict routes through `reject_policy.unclear`, which would send
    # every row of a caption-less corpus to the review queue.
    skip_requires: tuple[str, ...] = field(default_factory=tuple)
    # Whether the judge grades abstraction level 0. Off by default: AL0 is the
    # verbatim transcription of the graph, so its rubric ("every operator is
    # named", "operator order matches") is close to what the mechanical checks
    # already settle, and judging it costs a third of the run for the least
    # interesting level. The abstraction proper is AL1 and AL2.
    include_al0: bool = False

    def selected_criteria(self) -> tuple[Criterion, ...]:
        selected = self.criteria
        if self.criteria_filter:
            missing = [name for name in self.criteria_filter if all(name != item.id for item in selected)]
            if missing:
                raise ValueError("Unknown criteria %s." % ", ".join(missing))
            selected = tuple(item for item in selected if item.id in self.criteria_filter)
        if self.skip_requires:
            selected = tuple(item for item in selected if not set(item.requires) & set(self.skip_requires))
        if not selected:
            raise ValueError(
                "No criteria left to evaluate: criteria=%s, skip_requires=%s." % (
                    list(self.criteria_filter) or "all", list(self.skip_requires)
                )
            )
        return selected

    def dropped_criteria(self) -> tuple[Criterion, ...]:
        """Criteria excluded because this run cannot supply what they require."""
        if not self.skip_requires:
            return ()
        return tuple(item for item in self.criteria if set(item.requires) & set(self.skip_requires))

    def flag_names(self) -> tuple[str, ...]:
        return tuple(name for item in self.selected_criteria() for name in item.produces)

    @classmethod
    def from_config(cls, config_path: Path | str) -> "RejectionConfig":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if ROOT_KEY not in loaded:
            raise ValueError("Rejection config '%s' is missing a '%s' root key." % (path, ROOT_KEY))
        section = loaded[ROOT_KEY] or {}
        paths = section.get("paths") or {}
        run = section.get("run") or {}
        policy = section.get("reject_policy") or {}

        max_attempts = int(run.get("max_attempts", 3))
        if max_attempts < 1:
            raise ValueError("Rejection config '%s' sets run.max_attempts to %d; it must be at least 1." % (path, max_attempts))
        max_workers = int(run.get("max_workers", 4))
        if max_workers < 1:
            raise ValueError("Rejection config '%s' sets run.max_workers to %d; it must be at least 1." % (path, max_workers))

        criteria = tuple(_parse_criterion(entry, path) for entry in section.get("criteria") or [])
        if not criteria:
            raise ValueError("Rejection config '%s' declares no criteria." % path)

        unclear = section.get("reject_policy", {}).get("unclear") or {}
        if not isinstance(unclear, Mapping):
            raise ValueError("Rejection config '%s' must give reject_policy.unclear as a mapping of criterion to action." % path)
        for name, action in unclear.items():
            if action not in UNCLEAR_ACTIONS:
                raise ValueError("Rejection config '%s' sets reject_policy.unclear.%s to '%s'; expected one of %s." % (path, name, action, ", ".join(UNCLEAR_ACTIONS)))
        on_judge_error = policy.get("on_judge_error", DISPOSITION_REVIEW)
        if on_judge_error not in UNCLEAR_ACTIONS:
            raise ValueError("Rejection config '%s' sets reject_policy.on_judge_error to '%s'; expected one of %s." % (path, on_judge_error, ", ".join(UNCLEAR_ACTIONS)))

        config = cls(
            model=ModelSettings.from_mapping(section.get("model") or {}, str(path)),
            prompts_path=_required_path(paths, "prompts", path),
            analysis_path=_optional_path(paths, "analysis"),
            output_path=_required_path(paths, "output", path),
            review_path=_optional_path(paths, "review"),
            failures_path=_optional_path(paths, "failures"),
            full_manifest_path=_optional_path(paths, "full_manifest"),
            report_path=_optional_path(paths, "report"),
            captions_path=_optional_path(paths, "captions"),
            mert=MertSimilarityConfig.from_mapping(section.get("mert_similarity") or {}, str(path)),
            max_attempts=max_attempts,
            max_workers=max_workers,
            criteria=criteria,
            criteria_filter=tuple(str(name) for name in run.get("criteria") or []),
            subsample_num=int(run.get("subsample_num", -1)),
            seed=int(run.get("seed", 0)),
            reject_verdicts={
                name: tuple(values)
                for name, values in policy.items()
                if name not in ("unclear", "on_judge_error") and not name.endswith("_failures")
            },
            unclear_actions=dict(unclear),
            on_judge_error=on_judge_error,
            output_contract=str(section.get("output_contract") or ""),
            constraint_glosses=section.get("constraint_glosses") or {},
            caption_cues={
                name: tuple(str(cue).lower() for cue in cues)
                for name, cues in (section.get("caption_cues") or {}).items()
            },
            leak_syntax_patterns=tuple(str(item) for item in (section.get("implementation_leak") or {}).get("syntax_patterns") or []),
            leak_allow=tuple(str(item) for item in (section.get("implementation_leak") or {}).get("allow") or []),
            reject_failures={
                name[: -len("_failures")]: tuple(str(v) for v in values)
                for name, values in policy.items()
                if name.endswith("_failures")
            },
            version=int(loaded.get("version", 1))
        )
        _validate_rule_paths(config, path)
        return config


def load_rejection_config(
    config_dir: Path | str,
    rejection_config: Path | str | None = None
) -> RejectionConfig:
    """The rejection config from a config directory, or an explicit path."""
    if rejection_config is not None:
        return RejectionConfig.from_config(rejection_config)
    return RejectionConfig.from_config(Path(config_dir) / CONFIG_FILE_NAME)


def _parse_criterion(entry: Mapping[str, Any], config_path: Path) -> Criterion:
    if "id" not in entry:
        raise ValueError("Rejection config '%s' has a criterion with no id." % config_path)
    rules: list[Rule] = []
    for kind, key in (("joint", "rules_detail"), ("axis", "axis_rules"), ("era", "era_signatures")):
        for raw in entry.get(key) or []:
            rules.append(_parse_rule(raw, kind, config_path, entry["id"]))
    return Criterion(
        id=str(entry["id"]),
        produces=tuple(str(name) for name in entry.get("produces") or [entry["id"]]),
        per=str(entry.get("per", "row")),
        output=str(entry.get("output", "enum")),
        default_verdict=str(entry.get("default_verdict", "")),
        system_prompt=str(entry.get("system_prompt") or ""),
        rules=tuple(str(rule) for rule in entry.get("rules") or []),
        exemplars=tuple(entry.get("exemplars") or []),
        max_tokens=int(entry.get("max_tokens", 512)),
        values=tuple(str(value) for value in entry.get("values") or []),
        requires=tuple(str(name) for name in entry.get("requires") or []),
        rule_set=tuple(rules)
    )


def _parse_rule(raw: Mapping[str, Any], kind: str, config_path: Path, criterion_id: str) -> Rule:
    for required in ("id", "statement", "source"):
        if not raw.get(required):
            raise ValueError("Rejection config '%s' has a %s rule under '%s' missing '%s'. A rule without a source is a hunch." % (config_path, kind, criterion_id, required))
    return Rule(
        id=str(raw["id"]),
        statement=str(raw["statement"]),
        source=str(raw["source"]),
        kind=kind,
        trigger=raw.get("trigger") or raw.get("detect"),
        suggests=raw.get("suggests"),
        applies_to=raw.get("applies_to") or {},
        signature=raw.get("signature"),
        periods=tuple(str(value) for value in raw.get("periods") or []),
        genres=tuple(str(value) for value in raw.get("genres") or [])
    )


def _validate_rule_paths(config: RejectionConfig, config_path: Path) -> None:
    """Reject paths the context could never supply.

    `evaluate_condition` answers `False` for a path that does not resolve, so a
    misspelled path yields a rule that silently never fires and is
    indistinguishable from one that is correctly dormant. This is the check that
    keeps those apart, and it is why it runs at load rather than per row.

    It deliberately validates only the SHAPE of a path -- its root, and the
    aggregation under `params`. Whether the operator or parameter a path names
    actually exists is a question about reachability, not about the config:
    `apply_flanger_effect` and a filter `resonance` are named on purpose by
    rules that are meant to be dormant.
    """
    problems: list[str] = []
    for criterion in config.criteria:
        for rule in criterion.rule_set:
            if rule.trigger is None:
                problems.append("%s: no trigger or detect" % rule.id)
                continue
            for path in rule.paths:
                segments = path.split(".")
                if segments[0] not in CONTEXT_ROOTS:
                    problems.append("%s: path '%s' has unknown root '%s'; expected one of %s" % (rule.id, path, segments[0], ", ".join(CONTEXT_ROOTS)))
                    continue
                if segments[0] != "params":
                    continue
                if len(segments) < 2 or segments[1] not in PARAM_SCOPES:
                    problems.append("%s: path '%s' needs a scope from %s after 'params'" % (rule.id, path, ", ".join(PARAM_SCOPES)))
                    continue
                depth = 4 if segments[1] in PARAM_AGGREGATES or segments[1] == "label" else 5
                if len(segments) != depth:
                    problems.append("%s: path '%s' should have %d segments, has %d" % (rule.id, path, depth, len(segments)))
    if problems:
        raise ValueError("Rejection config '%s' has unusable rule paths:\n  %s" % (config_path, "\n  ".join(problems)))


def _required_path(paths: Mapping[str, Any], key: str, config_path: Path) -> Path:
    value = paths.get(key)
    if not value:
        raise ValueError("Rejection config '%s' is missing paths.%s." % (config_path, key))
    return Path(str(value)).expanduser()


def _optional_path(paths: Mapping[str, Any], key: str) -> Path | None:
    value = paths.get(key)
    if not value:
        return None
    return Path(str(value)).expanduser()


# --------------------------------------------------------------------------
# The trigger context
# --------------------------------------------------------------------------


def load_captions(path: Path) -> dict[str, str]:
    """MusicCaps captions by clip id, read from the dataset CSV.

    The prose caption is what this criterion is supposed to read, and it is the
    one thing the pipeline drops: the manifest keeps `aspect_list`, the mined
    phrase set, but `caption` arrives null in every prompt row. The two are not
    interchangeable. The phrases for one clip are

        slide guitar, blues, intricate acoustic guitar playing, guitar layers

    while the caption for the same clip says

        "This is a high-octane blues song ... There's a distant filtered vocal
         hum which sounds like it is coming from another room"

    Only the caption carries "distant" and "from another room" -- production
    language, which is precisely what the space and fidelity cues match on.
    Judging against the phrase list means judging against a description with the
    production stripped out of it.
    """
    import csv

    captions: dict[str, str] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "ytid" not in (reader.fieldnames or []) or "caption" not in (reader.fieldnames or []):
            raise ValueError("Captions file '%s' needs `ytid` and `caption` columns; found %s." % (path, reader.fieldnames))
        for row in reader:
            ytid, caption = row.get("ytid"), row.get("caption")
            if ytid and caption:
                captions[str(ytid)] = str(caption)
    return captions


def caption_text(
    row: Mapping[str, Any],
    analysis: Mapping[str, Any] | None = None,
    caption: str | None = None
) -> str:
    """The clip's caption plus its aspect phrases, as one lowercase string.

    Both halves matter. The caption is prose about the clip; the aspect list is
    the phrase set the tagger mined it into, and it survives distinctions the
    caption buries.

    Three places are read, because which of them is populated varies by how the
    artifact was built. Newer prompt rows carry the whole analysis record at
    `metadata.analysis`, which is the richest source and needs no re-join --
    and on the MusicCaps prompt artifact it is the ONLY source, since
    `metadata.caption` is null on every row there while
    `metadata.analysis.dataset.aspect_list` is populated on all of them. Older
    plan rows flatten the caption to a top-level `metadata.caption` and drop the
    aspects entirely, which is what the re-joined `analysis` argument covers.

    Reading all three and de-duplicating is cheaper than deciding which one a
    given artifact used.
    """
    metadata = row.get("metadata") or {}
    pieces: list[str] = []
    # The real prose caption first, where one was supplied, so it leads the
    # string the cues and the judge read.
    if caption:
        pieces.append(str(caption))
    # The plan row flattens this to a top-level `caption`, not `analysis.caption`.
    for candidate in (metadata.get("caption"), row.get("caption")):
        if candidate:
            pieces.append(str(candidate))
    # The row's own analysis first, then a re-joined one. Same shape either way.
    for record in (metadata.get("analysis"), analysis):
        if not record:
            continue
        if record.get("caption"):
            pieces.append(str(record["caption"]))
        dataset = record.get("dataset") or {}
        for aspect in dataset.get("aspect_list") or []:
            pieces.append(str(aspect))
    return " ".join(dict.fromkeys(pieces)).lower()


def caption_flags(text: str, cues: Mapping[str, Sequence[str]]) -> dict[str, bool]:
    """Which cue groups the caption text matches, by whole-word run.

    Phrases are matched on a word-boundary run rather than as substrings, so
    `mono` does not fire on "monotonous" and `old` does not fire on "golden".
    """
    words = tuple(word for word in WORD_RE.split(text) if word)
    joined = " %s " % " ".join(words)
    flags: dict[str, bool] = {}
    for name, phrases in cues.items():
        hit = False
        for phrase in phrases:
            needle = " %s " % " ".join(word for word in WORD_RE.split(phrase.lower()) if word)
            if needle.strip() and needle in joined:
                hit = True
                break
        flags[name] = hit
    return flags


def param_maps(reader: ChainReader, graph_spec: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The `params` context root: aggregates and block scopes over a graph.

    The condition DSL has no wildcard and no list indexing, so a rule cannot
    itself ask "does any occurrence of this operator exceed X". Collapsing
    repeats to a max and a min up front is what lets `gte` and `lte` mean that.
    Block scopes come along because some rules must distinguish a pass filter on
    a reverb's wet return from one narrowing the signal itself.
    """
    aggregates: dict[str, dict[str, dict[str, float]]] = {"max": {}, "min": {}}
    scoped: dict[str, dict[str, dict[str, dict[str, float]]]] = {
        "send": {"max": {}, "min": {}},
        "chain": {"max": {}, "min": {}}
    }
    labelled: dict[str, dict[str, Any]] = {}

    def record(bucket: dict[str, dict[str, float]], operator: str, param: str, value: float, keep_larger: bool) -> None:
        current = bucket.setdefault(operator, {})
        if param not in current:
            current[param] = value
            return
        current[param] = max(current[param], value) if keep_larger else min(current[param], value)

    for chain_param in reader.iter_params(graph_spec):
        labelled.setdefault(chain_param.label, {})[chain_param.param] = chain_param.value
        if not isinstance(chain_param.value, (int, float)) or isinstance(chain_param.value, bool):
            continue
        value = float(chain_param.value)
        record(aggregates["max"], chain_param.operator, chain_param.param, value, True)
        record(aggregates["min"], chain_param.operator, chain_param.param, value, False)
        scope = "send" if chain_param.block_kind == "send_return" else "chain" if chain_param.block_kind == "chain" else None
        if scope is not None:
            record(scoped[scope]["max"], chain_param.operator, chain_param.param, value, True)
            record(scoped[scope]["min"], chain_param.operator, chain_param.param, value, False)

    return {
        "max": aggregates["max"],
        "min": aggregates["min"],
        "label": labelled,
        "send": scoped["send"],
        "chain": scoped["chain"]
    }


def derived_fields(row: Mapping[str, Any], params: Mapping[str, Any], analysis: Mapping[str, Any] | None) -> dict[str, Any]:
    """Values no single parameter carries, recovered from the ones that do.

    `delay_beats` is the notable one. The tempo_sync motif resolves
    `delay_seconds = (60 / bpm) * beats` and only the product reaches the graph,
    so the note division -- which is what carries a genre signature -- has to be
    inverted back out. The bpm here coalesces exactly as motifs.yaml does, so
    the inversion is faithful as long as the manifest still reports the tempo
    the planner saw.
    """
    metadata = row.get("metadata") or {}
    tempo = (analysis or {}).get("tempo_bpm")
    if tempo is None:
        tempo = metadata.get("tempo_bpm")
    bpm = float(tempo) if tempo else DEFAULT_TEMPO_BPM
    seconds = ((params.get("max") or {}).get("apply_delay_effect") or {}).get("delay_seconds")
    fields: dict[str, Any] = {
        "tempo_bpm": bpm,
        "tempo_known": tempo is not None,
        "has_caption": False
    }
    if seconds is not None:
        fields["delay_beats"] = round(float(seconds) * bpm / 60.0, 4)
    return fields


def build_context(
    row: Mapping[str, Any],
    reader: ChainReader,
    config: RejectionConfig,
    analysis: Mapping[str, Any] | None = None,
    caption: str | None = None
) -> dict[str, Any]:
    """The context a rule's trigger is evaluated against.

    Deliberately not the planner's context. That one is built from an analysis
    manifest record and exposes `metadata`/`bindings`/`pattern`; by the time a
    row reaches this stage the manifest has been flattened and the graph
    resolved, so the useful roots are different ones. Rules here address
    `params`, `plan`, `profile`, `caption`, `analysis` and `derived`.
    """
    metadata = row.get("metadata") or {}
    graph_spec = metadata.get("graph_spec") or []
    profile = reader.profile(graph_spec)
    params = param_maps(reader, graph_spec)
    text = caption_text(row, analysis, caption)
    flags = caption_flags(text, config.caption_cues)
    derived = derived_fields(row, params, analysis)
    derived["has_caption"] = bool(text.strip())
    return {
        "params": params,
        "plan": {
            "recipe_id": metadata.get("recipe_id"),
            "recipe_tags": list(metadata.get("recipe_tags") or []),
            "target_family": metadata.get("target_family"),
            "target_stem": metadata.get("target_stem"),
            "genres": list(metadata.get("genres") or []),
            "mood_themes": list(metadata.get("mood_themes") or []),
            "issues": list(metadata.get("issues") or []),
            "poison_tags": list(metadata.get("poison_tags") or []),
            "caption": text
        },
        "profile": {
            "tags": list(profile.tags),
            "operators": list(profile.operators),
            "magnitude": profile.magnitude.to_dict()
        },
        "caption": flags,
        "analysis": dict(analysis or {}),
        "derived": derived
    }


def triggered_rules(criterion: Criterion, context: Mapping[str, Any]) -> tuple[Rule, ...]:
    """The criterion's rules whose trigger fires on this context."""
    return tuple(
        rule
        for rule in criterion.rule_set
        if rule.trigger is not None and evaluate_condition(rule.trigger, context)
    )


# --------------------------------------------------------------------------
# Reachability
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleReachability:
    """Whether a rule's trigger can fire at all under the current configs."""

    rule_id: str
    criterion_id: str
    kind: str
    verdict: str
    detail: str
    source: str


def support_minimum(support: Support) -> float:
    candidates = list(support.points)
    candidates.extend(low for low, _ in support.intervals)
    return min(candidates)


def reachable_operators(motifs: Mapping[str, Any], recipes: Any) -> set[str]:
    """Operators any motif or recipe can actually place in a graph.

    Not the same set as operators.yaml, which is what may be DECLARED. Five
    declared operators are referenced by neither file, so a rule about them can
    never fire however wide its parameter supports are -- and that is a
    different kind of dormancy from a narrow distribution, because widening a
    distribution cannot fix it.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            operator = node.get("operator")
            if isinstance(operator, str):
                found.add(operator)
            # A `separate` or `mix` block names no operator and gets one by
            # default, so scanning for `operator:` keys alone would report the
            # two routing operators as never placed.
            kind = node.get("kind")
            if isinstance(kind, str) and kind in BLOCK_DEFAULT_OPERATORS:
                found.add(BLOCK_DEFAULT_OPERATORS[kind])
            for value in node.values():
                walk(value)
            return
        if isinstance(node, (list, tuple)):
            for value in node:
                walk(value)

    walk(motifs)
    walk(recipes)
    return found


def _param_support(
    lexicon: BandLexicon,
    distributions: Mapping[str, Any],
    literals: Mapping[str, Support],
    operator: str,
    param: str,
    is_block: bool
) -> Support | None:
    spec = lexicon.block_spec(operator, param) if is_block else lexicon.param_spec(operator, param)
    if spec is None:
        return None
    support = Support()
    for ref in spec.sample_refs:
        # `param` is required: a `joint` prior models several parameters
        # together, so its support depends on which one is being asked about.
        support = support.merge(BandLexicon._distribution_support(ref, distributions, param))
    literal = literals.get("%s.%s" % (operator, param))
    if literal is not None:
        support = support.merge(literal)
    return support if not support.is_empty() else None


def rule_reachability(
    rule: Rule,
    criterion_id: str,
    lexicon: BandLexicon,
    registry: OperatorRegistry,
    distributions: Mapping[str, Any],
    literals: Mapping[str, Support],
    operators_in_use: set[str]
) -> RuleReachability:
    """One rule's reachability, derived rather than declared.

    Two gates, operator first. A trigger over `params` names an operator, and if
    no motif or recipe can place that operator the parameter supports are
    irrelevant. Only then does the region the trigger selects get intersected
    against what the samplers can actually produce.

    Leaves of an `all` are treated independently, which makes `live` a sound
    over-approximation: leaves that are individually reachable might still be
    jointly unsatisfiable. That errs toward reporting a rule live, which is the
    safe direction -- a rule wrongly called dormant would be quietly ignored.
    """
    named = _named_operators(rule.trigger)
    unplaceable = [
        name for name in named
        if not registry.has(name) or name not in operators_in_use
    ]
    if unplaceable:
        detail = "; ".join(
            "operator '%s' is %s" % (
                name,
                "not declared in operators.yaml" if not registry.has(name) else "declared but referenced by no motif or recipe"
            )
            for name in unplaceable
        )
        return RuleReachability(rule.id, criterion_id, rule.kind, REACH_UNREACHABLE_OPERATOR, detail, rule.source)

    leaves = _trigger_leaves(rule.trigger)
    param_leaves = [leaf for leaf in leaves if leaf[0].startswith("params.")]
    if not param_leaves:
        detail = "trigger reads only caption or plan metadata, so reachability depends on the corpus rather than the configs"
        return RuleReachability(rule.id, criterion_id, rule.kind, REACH_DATA_DEPENDENT, detail, rule.source)

    verdicts: list[tuple[str, str]] = []
    for path, key, value in param_leaves:
        segments = path.split(".")
        scope = segments[1]
        operator, param = (segments[2], segments[3]) if scope in PARAM_AGGREGATES or scope == "label" else (segments[3], segments[4])
        if scope == "label":
            verdicts.append((REACH_DATA_DEPENDENT, "'%s' is scoped to a graph step label" % path))
            continue
        if operator != "send_return" and not registry.has(operator):
            verdicts.append((REACH_UNREACHABLE_OPERATOR, "operator '%s' is not declared in operators.yaml" % operator))
            continue
        if operator != "send_return" and operator not in operators_in_use:
            verdicts.append((REACH_UNREACHABLE_OPERATOR, "operator '%s' is declared but referenced by no motif or recipe" % operator))
            continue
        support = _param_support(lexicon, distributions, literals, operator, param, operator == "send_return")
        if support is None:
            verdicts.append((REACH_DORMANT_NO_PARAM, "'%s.%s' has no sampled or pinned support" % (operator, param)))
            continue
        low, high = COMPARISON_REGIONS[key](float(value))
        region_low = float("-inf") if low is None else float(low)
        region_high = float("inf") if high is None else float(high)
        if key in ("gte", "gt"):
            region_high = float("inf")
        if not support.intersects(region_low, region_high if region_high != region_low else region_high + 1e-9):
            verdicts.append((REACH_DORMANT_DISTRIBUTION, "'%s.%s' samples at most %s, never %s %s" % (operator, param, support.maximum(), key, value)))
            continue
        whole = support_minimum(support) >= region_low and support.maximum() <= region_high
        if whole:
            verdicts.append((REACH_VACUOUS, "'%s.%s' is always %s %s (support %s..%s), so this leaf is constant" % (operator, param, key, value, support_minimum(support), support.maximum())))
            continue
        verdicts.append((REACH_LIVE, "'%s.%s' can be %s %s" % (operator, param, key, value)))

    # An `all` is dormant if any leaf is, and vacuous only if every leaf is.
    order = (REACH_UNREACHABLE_OPERATOR, REACH_DORMANT_NO_PARAM, REACH_DORMANT_DISTRIBUTION)
    for blocking in order:
        blocked = [detail for verdict, detail in verdicts if verdict == blocking]
        if blocked:
            return RuleReachability(rule.id, criterion_id, rule.kind, blocking, "; ".join(blocked), rule.source)
    if verdicts and all(verdict == REACH_VACUOUS for verdict, _ in verdicts):
        return RuleReachability(rule.id, criterion_id, rule.kind, REACH_VACUOUS, "; ".join(detail for _, detail in verdicts), rule.source)
    vacuous = [detail for verdict, detail in verdicts if verdict == REACH_VACUOUS]
    detail = "; ".join(detail for _, detail in verdicts)
    if vacuous:
        detail = "%s (note: %d leaf/leaves always true)" % (detail, len(vacuous))
    return RuleReachability(rule.id, criterion_id, rule.kind, REACH_LIVE, detail, rule.source)


def _named_operators(condition: Mapping[str, Any] | None) -> list[str]:
    """Operator names a condition tests membership of `profile.operators` for.

    A rule can select an effect either by constraining one of its parameters or
    by asking whether the operator is in the chain at all. The second form
    carries the same reachability question as the first, and reading it out of
    a `contains_any` is what keeps a rule about an operator no recipe places
    from being reported as merely data-dependent.
    """
    if condition is None:
        return []
    names: list[str] = []
    for key, payload in condition.items():
        if key in ("all", "any", "not"):
            nested = payload if isinstance(payload, list) else [payload]
            for item in nested:
                names.extend(_named_operators(item))
            continue
        if key in ("contains_any", "contains_all", "in") and isinstance(payload, Mapping):
            if payload.get("path") == "profile.operators":
                names.extend(str(value) for value in payload.get("values") or [])
    return names


def _trigger_leaves(condition: Mapping[str, Any] | None) -> list[tuple[str, str, Any]]:
    """Every `(path, comparison_key, value)` a condition compares on."""
    if condition is None:
        return []
    leaves: list[tuple[str, str, Any]] = []
    for key, payload in condition.items():
        if key in ("all", "any", "not"):
            nested = payload if isinstance(payload, list) else [payload]
            for item in nested:
                leaves.extend(_trigger_leaves(item))
            continue
        if key in COMPARISON_REGIONS and isinstance(payload, Mapping) and "path" in payload:
            leaves.append((str(payload["path"]), key, payload.get("value")))
    return leaves


def reachability_report(
    config: RejectionConfig,
    lexicon: BandLexicon,
    registry: OperatorRegistry,
    distributions: Mapping[str, Any],
    motifs: Mapping[str, Any],
    recipes: Any
) -> tuple[RuleReachability, ...]:
    """Every rule's reachability, in config order."""
    literals = literal_param_support(motifs, recipes)
    operators_in_use = reachable_operators(motifs, recipes)
    return tuple(
        rule_reachability(rule, criterion.id, lexicon, registry, distributions, literals, operators_in_use)
        for criterion in config.criteria
        for rule in criterion.rule_set
    )


# --------------------------------------------------------------------------
# Prompt assembly
# --------------------------------------------------------------------------


def system_prompt_for(criterion: Criterion, config: RejectionConfig) -> str:
    """A criterion's voice plus the shared output contract."""
    parts = [criterion.system_prompt.strip()]
    if config.output_contract.strip():
        parts.append(config.output_contract.strip())
    return "\n\n".join(part for part in parts if part)


def render_level_prompt(
    row: Mapping[str, Any],
    entry: Mapping[str, Any],
    parent_text: str | None,
    criterion: Criterion,
    config: RejectionConfig,
    hard_violations: Sequence[str],
    level: Any = None
) -> str:
    """The judge prompt for one abstraction level.

    Carries the level's own rules and rubric, the contract it was written under
    glossed into obligations, and the graph or parent text it must stay true to.

    The rubric, rules and constraints come from the CURRENT ladder when one is
    supplied, not from the row. The row's copies were frozen at generation time,
    so judging against them would grade every artifact by whatever the ladder
    happened to say when it was written -- and an edit to a rubric would never
    reach the judge. Same reason the mechanical checks are re-run rather than
    read back. The row's copies remain the fallback for a level the ladder no
    longer defines.
    """
    blocks: list[str] = []
    blocks.append("EDIT GRAPH\n%s" % (row.get("metadata", {}).get("graph_description") or "(none recorded)"))
    if parent_text is None:
        blocks.append("SOURCE\nThis level is written from the edit graph above, so the graph is its source.")
    else:
        blocks.append("SOURCE (abstraction level %s)\n%s" % (entry.get("derived_from"), parent_text))
    blocks.append("INSTRUCTION UNDER AUDIT (abstraction level %s, '%s')\n%s" % (entry.get("abstraction_level"), entry.get("name"), entry.get("text") or ""))

    rules = list(getattr(level, "rules", None) or entry.get("rules") or [])
    if rules:
        blocks.append("RULES THIS INSTRUCTION WAS WRITTEN UNDER\n%s" % "\n".join("- %s" % rule for rule in rules))

    rubric = list(getattr(level, "rubric", None) or entry.get("rubric") or [])
    if rubric:
        blocks.append("RUBRIC ITEMS, BY INDEX\n%s" % "\n".join("%d. %s" % (index, item) for index, item in enumerate(rubric)))

    constraints = getattr(level, "constraints", None) or entry.get("constraints") or {}
    glossed = render_constraint_glosses(constraints, config)
    if glossed:
        # Retitled from "CONTRACT": under the fidelity framing this is no
        # longer the thing being graded, it is what makes "left something out"
        # answerable at all. AL2 is *supposed* to drop every parameter value,
        # so a judge asked about omission without this would fail every level.
        blocks.append(
            "WHAT THIS LEVEL IS LICENSED TO DISCARD\n%s\n"
            "Discarding any of the above is correct and is never an omission. "
            "Omission means dropping something this level was obliged to keep." % glossed
        )

    bands = row.get("band_assignments") or {}
    if bands:
        blocks.append("BAND THE GRAPH SPECIFIES FOR EACH PARAMETER\n%s" % "\n".join("- %s: %s" % (name, band) for name, band in sorted(bands.items())))

    magnitude = row.get("chain_magnitude") or {}
    # Omitted when the chain has no magnitude-role parameter at all: printing
    # "None (None)" tells the judge nothing and invites it to invent a reading.
    if magnitude.get("band") is not None:
        blocks.append("COMPUTED CHAIN INTENSITY\n%s (%s)" % (magnitude.get("band"), magnitude.get("value")))

    tags = row.get("chain_tags") or []
    if tags:
        blocks.append("CHAIN CHARACTER TAGS\n%s" % ", ".join(tags))

    if hard_violations:
        failed = "\n".join("- %s" % item for item in hard_violations)
        blocks.append("MECHANICAL CHECKS ALREADY FAILED\n%s\nThese are settled. Judge only what they cannot cover." % failed)

    blocks.append(render_criterion_rules(criterion))
    blocks.append(render_exemplars(criterion))
    blocks.append(
        "Return JSON of exactly this shape:\n"
        '{"al_rules": {"verdict": "pass|fail", "failed_rubric_items": [<int>], '
        '"rationale": "<ONE sentence: which rubric item is broken and how>"}, '
        '"al_consistency": {"verdict": "pass|fail", "failures": [%s], '
        '"rationale": "<ONE sentence: what it gets wrong about the source>"}}\n'
        "Give each verdict its own rationale. They answer different questions and "
        "can disagree, so one sentence covering both explains neither. A passing "
        "verdict needs no rationale and an empty failures list."
        % " | ".join('"%s"' % item for item in CONSISTENCY_FAILURES)
    )
    return "\n\n".join(block for block in blocks if block)


def render_constraint_glosses(constraints: Mapping[str, Any], config: RejectionConfig) -> str:
    """A level's declared constraints as the obligations they imply.

    The ladder already states what each level may change; this turns those
    declarations into something judgeable without restating them as prose rules
    somewhere else, so a level added to the ladder is covered as soon as its
    constraint values have glosses.
    """
    lines: list[str] = []
    for key, value in sorted(constraints.items()):
        gloss = (config.constraint_glosses.get(key) or {}).get(str(value))
        if gloss:
            lines.append("- %s = %s: %s" % (key, value, " ".join(str(gloss).split())))
        else:
            lines.append("- %s = %s" % (key, value))
    return "\n".join(lines)


def render_criterion_rules(criterion: Criterion) -> str:
    if not criterion.rules:
        return ""
    return "HOW TO JUDGE\n%s" % "\n".join("- %s" % rule for rule in criterion.rules)


def render_exemplars(criterion: Criterion) -> str:
    if not criterion.exemplars:
        return ""
    lines: list[str] = []
    for exemplar in criterion.exemplars:
        lines.append("- %s: %s" % (exemplar.get("verdict"), " ".join(str(exemplar.get("note", "")).split())))
    return "CALIBRATION EXAMPLES\n%s" % "\n".join(lines)


def render_rule_evidence(rules: Sequence[Rule], context: Mapping[str, Any]) -> str:
    """The triggered rules, their sources, and the values that tripped them."""
    lines: list[str] = []
    for rule in rules:
        lines.append("RULE %s" % rule.id)
        if rule.signature:
            lines.append("  signature: %s" % rule.signature)
        lines.append("  statement: %s" % " ".join(rule.statement.split()))
        lines.append("  source: %s" % rule.source)
        if rule.suggests:
            lines.append("  this rule alone would suggest: %s" % rule.suggests)
        if rule.periods or rule.genres:
            lines.append("  belongs to periods [%s] and genres [%s]" % (", ".join(rule.periods) or "any", ", ".join(rule.genres) or "any"))
        for path, key, value in _trigger_leaves(rule.trigger):
            actual = _safe_lookup(context, path)
            lines.append("  measured: %s = %s (rule fires at %s %s)" % (path, actual, key, value))
    return "\n".join(lines)


def _safe_lookup(context: Mapping[str, Any], path: str) -> Any:
    node: Any = context
    for segment in path.split("."):
        if not isinstance(node, Mapping) or segment not in node:
            return None
        node = node[segment]
    return node


def render_row_prompt(
    row: Mapping[str, Any],
    criterion: Criterion,
    config: RejectionConfig,
    context: Mapping[str, Any],
    rules: Sequence[Rule]
) -> str:
    """The judge prompt for a row-level criterion with rules already fired."""
    metadata = row.get("metadata") or {}
    blocks: list[str] = []
    caption = context.get("plan", {}).get("caption")
    if caption:
        blocks.append("HOW THE MUSIC IS DESCRIBED\n%s" % caption)
    genres = context.get("plan", {}).get("genres") or []
    blocks.append("MINED GENRE TAGS\n%s" % (", ".join(genres) if genres else "(none mined; the caption is all there is to go on)"))
    blocks.append("EDIT GRAPH\n%s" % (metadata.get("graph_description") or "(none recorded)"))
    blocks.append("TARGET\n%s stem (family %s), recipe %s" % (metadata.get("target_stem"), metadata.get("target_family"), metadata.get("recipe_id")))
    magnitude = row.get("chain_magnitude") or {}
    if magnitude:
        blocks.append("COMPUTED CHAIN INTENSITY\n%s (%s)" % (magnitude.get("band"), magnitude.get("value")))
    blocks.append("RULES THAT FIRED\n%s" % render_rule_evidence(rules, context))
    blocks.append(render_criterion_rules(criterion))
    blocks.append(render_exemplars(criterion))
    blocks.append(
        "Return JSON of exactly this shape:\n"
        '{"verdict": "%s", "cited_rules": ["<rule id>"], "caption_evidence": "<the words you relied on, or null>", '
        '"rationale": "<ONE sentence saying what is wrong and why>"}'
        % "|".join(criterion.values)
    )
    return "\n\n".join(block for block in blocks if block)


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def parse_verdict(text: str, criterion: Criterion) -> tuple[dict[str, Any] | None, tuple[str, ...]]:
    """A judge response as a dict, plus any reasons it is unusable.

    Lenient about packaging and strict about content: models wrap JSON in prose
    and fences however they like, but a verdict outside the declared vocabulary
    is refused rather than coerced, so a parse failure can never read as a pass.
    """
    if not text or not text.strip():
        return None, ("empty response",)
    match = JSON_BLOCK_RE.search(text)
    if match is None:
        return None, ("no JSON object in the response",)
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return None, ("response was not valid JSON (%s)" % error.msg,)
    if not isinstance(parsed, Mapping):
        return None, ("response JSON was not an object",)

    problems: list[str] = []
    if criterion.output == "binary":
        for flag in criterion.produces:
            section = parsed.get(flag)
            if not isinstance(section, Mapping):
                problems.append("missing object '%s'" % flag)
                continue
            if section.get("verdict") not in ("pass", "fail"):
                problems.append("'%s.verdict' must be pass or fail, got %r" % (flag, section.get("verdict")))
            # A judge inventing a fourth failure label is a parse failure, not
            # a silently-dropped field: the retry gets a chance to fix it.
            failures = section.get("failures")
            if failures is not None:
                if not isinstance(failures, list):
                    problems.append("'%s.failures' must be a list" % flag)
                else:
                    unknown = [item for item in failures if item not in CONSISTENCY_FAILURES]
                    if unknown:
                        problems.append("'%s.failures' has unknown value(s) %s; expected from %s" % (flag, unknown, ", ".join(CONSISTENCY_FAILURES)))
    else:
        if parsed.get("verdict") not in criterion.values:
            problems.append("'verdict' must be one of %s, got %r" % (", ".join(criterion.values), parsed.get("verdict")))
    if problems:
        return None, tuple(problems)
    return dict(parsed), ()


def render_retry_note(problems: Sequence[str]) -> str:
    """The correction appended to a prompt after an unusable response.

    The generator keeps a nonzero temperature so a re-send differs; the judge
    runs at zero and relies on this note changing the input instead.
    """
    return "\n\nYour previous response was rejected because it %s. Return only the JSON object described above." % ", and ".join(problems)


def disposition_for(
    flags: Mapping[str, Mapping[str, Any]],
    config: RejectionConfig
) -> tuple[str, tuple[str, ...]]:
    """A row's disposition and the flags responsible.

    `unclear` is a configurable third outcome rather than a silent accept, and a
    judge error never resolves to one either.
    """
    reject_reasons: list[str] = []
    review_reasons: list[str] = []
    for name, flag in flags.items():
        verdict = flag.get("verdict")
        if flag.get("judge_error"):
            if config.on_judge_error == "reject":
                reject_reasons.append(name)
            elif config.on_judge_error == "review":
                review_reasons.append(name)
            continue
        # A flag carrying `failures` is rejected on which kinds it found, not
        # on the bare verdict: the verdict says the rewrite is wrong, the
        # failure types say whether that kind of wrongness disqualifies it.
        rejecting_failures = config.reject_failures.get(name)
        if rejecting_failures is not None and "failures" in flag:
            if set(flag.get("failures") or ()) & set(rejecting_failures):
                reject_reasons.append(name)
            continue
        if verdict in config.reject_verdicts.get(name, ()):
            reject_reasons.append(name)
            continue
        if verdict in ("unclear", None):
            action = config.unclear_actions.get(name, config.unclear_actions.get("default", "keep"))
            if action == "reject":
                reject_reasons.append(name)
            elif action == "review":
                review_reasons.append(name)
    if reject_reasons:
        return DISPOSITION_REJECT, tuple(reject_reasons)
    if review_reasons:
        return DISPOSITION_REVIEW, tuple(review_reasons)
    return DISPOSITION_ACCEPT, ()


# Keys that identify a row written by scripts/generate_agent_input.py: it nests
# the whole prompt row under `metadata` and adds the two audio paths beside it.
AGENT_INPUT_KEYS = ("ground_truth_edit_audio", "input_audio", "metadata")

# Keys a prompt row carries at its own top level.
PROMPT_ROW_KEYS = ("clip_id", "plan_id", "prompt_levels")


def is_agent_input(row: Mapping[str, Any]) -> bool:
    """Whether a row is agent input rather than a bare prompt row.

    Decided on the nesting, not on the audio keys alone: a prompt row also
    carries `input_audio`, so only the whole prompt row sitting under
    `metadata` distinguishes the two.
    """
    if not all(key in row for key in AGENT_INPUT_KEYS):
        return False
    nested = row.get("metadata")
    return isinstance(nested, Mapping) and all(key in nested for key in PROMPT_ROW_KEYS)


@dataclass(frozen=True)
class RowAudio:
    """The rendered pair a row points at, if it carries one."""

    reference_path: str
    edited_path: str
    reference: str
    poison_repair: bool


def normalize_row(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], RowAudio | None]:
    """A row as `(prompt_row, audio)`, accepting either input format.

    Agent input is the richer of the two and the default, because it is the only
    artifact carrying both the prompt and the location of the rendered audio --
    the prompt file has no paths, the render manifest has no prompts. Unwrapping
    it here means every criterion downstream sees the same prompt-row shape
    whichever file was passed.

    The reference side needs care. In the ordinary mode agent input sets
    `input_audio` to the untouched source, so the similarity measures the edit
    *plus* Demucs separation loss; the no-FX baseline that isolates the edit
    alone lives only in the render manifest, which is why a manifest can still
    be supplied to upgrade it.

    In poisoning mode agent input sets `input_audio` to the *degraded* audio and
    moves the clean original to `metadata.original_clean_audio`. Comparing
    against the degraded version would measure repair fidelity rather than edit
    strength, so the clean original is preferred where it exists.
    """
    if not is_agent_input(row):
        return row, None

    prompt_row = row["metadata"]
    edited = row.get("ground_truth_edit_audio")
    plan = prompt_row.get("metadata") or {}
    poison_repair = bool(plan.get("poison_graph_spec"))
    clean = prompt_row.get("original_clean_audio") or row.get("original_clean_audio")

    if poison_repair and clean:
        reference, kind = clean, "source"
    elif poison_repair:
        # No clean original recorded, so the only reference is the degraded
        # input. Still reported, but marked so it is never pooled with the rest.
        reference, kind = row.get("input_audio"), "poisoned_input"
    else:
        reference, kind = row.get("input_audio"), "source"

    if not reference or not edited:
        return prompt_row, None
    return prompt_row, RowAudio(str(reference), str(edited), kind, poison_repair)


def leak_identifiers(registry: OperatorRegistry, graph_spec: Sequence[Mapping[str, Any]]) -> set[str]:
    """Identifiers that would betray the implementation if they appeared verbatim.

    Derived rather than listed, from two sources: the operator registry, which
    supplies every function name and parameter name the pipeline can emit, and
    the row's own graph, which supplies the step labels and stem handles
    (`target_stem`, `residual`, `hum_cleanup`) that vary per plan. A new
    operator or a renamed motif step is therefore covered with no edit here.

    Only multi-word identifiers are returned -- anything carrying an underscore.
    A single bare word like `drums` or `reverb` is exactly what a human would
    say, so matching on it would flag correct prose; `apply_reverb_effect` and
    `room_size` are things only a machine writes.
    """
    identifiers: set[str] = set()
    for name in registry.names():
        spec = registry.resolve(name)
        identifiers.add(name)
        identifiers.update(spec.aliases)
        identifiers.update(spec.params)

    def walk(node: Any) -> None:
        if isinstance(node, Mapping):
            for field_name, value in node.items():
                if field_name in GRAPH_IDENTIFIER_FIELDS and isinstance(value, str):
                    identifiers.add(value)
                walk(value)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk(graph_spec)
    return {item for item in identifiers if "_" in item}


def implementation_leaks(
    text: str,
    identifiers: Sequence[str],
    syntax_patterns: Sequence[str]
) -> list[dict[str, str]]:
    """Every machine-only token or syntax fragment an instruction leaked.

    An instruction is meant to read as something a person would say. Text that
    names `separate_audio`, writes `apply_reverb_effect(room_size=0.9)` or
    carries a `->` from the graph is describing the implementation instead, and
    is not usable as training text whatever else is right about it.

    Matched on word boundaries against the raw text, not the normalized form the
    mention checks use: the underscores and punctuation are precisely the
    evidence here, so stripping them would destroy the signal.
    """
    if not text:
        return []
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for identifier in identifiers:
        if re.search(r"\b%s\b" % re.escape(identifier), text):
            key = ("identifier", identifier)
            if key not in seen:
                seen.add(key)
                found.append({"kind": "identifier", "token": identifier})
    for pattern in syntax_patterns:
        match = re.search(pattern, text)
        if match:
            key = ("syntax", pattern)
            if key not in seen:
                seen.add(key)
                found.append({"kind": "syntax", "token": match.group(0)[:40], "pattern": pattern})
    return found


def report_row(
    record: Mapping[str, Any],
    verdict: Mapping[str, Any],
    config: "RejectionConfig | None" = None
) -> dict[str, Any]:
    """One flat row: the prompts, their ids, and each flag's status.

    The lean counterpart to `full_manifest_row`. One column per flag rather than
    a nested object, so the result loads straight into a dataframe and a flag
    rate is a groupby rather than a traversal.

    `prompt_variants` is carried verbatim because it is what was actually
    judged: it is exactly `[e["text"] for e in prompt_levels]` ordered by level,
    so a reader can see the text a verdict refers to without opening the input.

    Carries judged verdicts only. The mechanical `checks.hard` results are
    deliberately absent: they are a regex's answer to a different question, and
    they are in the sidecar for anyone who needs them. `implementation_leak` is
    here despite also being deterministic, because it is a requested criterion
    in its own right rather than part of the ladder's hard checks.
    """
    flags = verdict.get("flags") or {}
    row: dict[str, Any] = {
        "clip_id": verdict.get("clip_id") or record.get("clip_id"),
        "plan_id": verdict.get("plan_id") or record.get("plan_id"),
        # What the stylistic criterion actually read, so a verdict about the
        # music can be checked against the description it was based on. On the
        # MusicCaps artifact the raw caption is null and this is the mined
        # aspect phrases; `caption_text` joins whichever is present.
        "caption": verdict.get("caption"),
        # The rendered graph, not `graph_spec`. This is the exact string the
        # judge was shown in its EDIT GRAPH block, so a verdict can be read
        # against what produced it; the structured form is five times the size
        # and is in the full manifest for anyone who needs to query it.
        "graph_description": (record.get("metadata") or {}).get("graph_description"),
        "prompt_variants": list(record.get("prompt_variants") or [])
    }
    for name, flag in flags.items():
        if name == "mechanical":
            # Left out of the report on purpose. `checks.hard` answers a
            # different question from the judge -- whether a regex matched --
            # and on this corpus it fails so often that it drowns the verdicts
            # this table exists to show. It is still computed and still written
            # to the sidecar, which is where to go when a judged verdict needs
            # explaining.
            continue
        if name == "implementation_leak":
            row["implementation_leak"] = flag.get("verdict")
            row["implementation_leak_levels"] = list(flag.get("leaked_levels") or [])
            # The offending tokens, deduplicated across levels. This is the
            # actionable part -- knowing a row leaked is less use than knowing
            # it leaked `separate_audio` and a graph arrow.
            tokens: list[str] = []
            for part in (flag.get("per_level") or {}).values():
                for leak in part.get("leaks") or []:
                    if leak["token"] not in tokens:
                        tokens.append(leak["token"])
            row["implementation_leak_tokens"] = tokens
            continue
        # A flag with no verdict reports its reason instead, so a null column is
        # never silently indistinguishable from a criterion that did not run.
        row[name] = flag.get("verdict")
        if flag.get("failures"):
            row["%s_failures" % name] = list(flag["failures"])
        reason = flag.get("reason") or next(
            (label for label in ("judge_error", "dry_run") if flag.get(label)), None
        )
        if flag.get("verdict") is None and reason:
            row["%s_reason" % name] = reason
        # The judge's one-sentence justification, but only where the verdict
        # actually rejects. A reason attached to a pass is noise in a table
        # nobody will read, and it doubles the width of the common case.
        #
        # Which verdicts count as failing comes from `reject_policy`, not a
        # hardcoded list, so adding `unclear` to a criterion's reject list makes
        # its reasons appear too. Without a config the rationale is carried
        # as-is, since there is then nothing to decide against.
        rejecting = config.reject_verdicts.get(name, ()) if config is not None else None
        fails = rejecting is None or flag.get("verdict") in rejecting
        if flag.get("rationale") and fails:
            row["%s_why" % name] = flag["rationale"]
    row["disposition"] = verdict.get("disposition")
    row["reject_reasons"] = list(verdict.get("reject_reasons") or [])
    return row


def full_manifest_row(
    record: Mapping[str, Any],
    verdict: Mapping[str, Any]
) -> dict[str, Any]:
    """One input record in full, with its verdict attached.

    The input keys are copied through untouched and the verdict lands under a
    single `rejection` key, so a consumer of the prompt schema keeps working and
    a training job needs no join. Nothing is dropped and nothing is renamed.
    """
    row = dict(record)
    row["rejection"] = {
        key: value
        for key, value in verdict.items()
        if key not in ("clip_id", "plan_id")
    }
    return row


def index_analysis(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Analysis-manifest records by clip id, for the re-join."""
    index: dict[str, Mapping[str, Any]] = {}
    for record in records:
        clip_id = record.get("clip_id")
        if clip_id is not None:
            index[str(clip_id)] = record.get("analysis") or {}
    return index


def iter_level_jobs(
    row: Mapping[str, Any],
    ladder: AbstractionLadder
) -> Iterator[tuple[Mapping[str, Any], str | None]]:
    """Each prompt level of a row, paired with the text it was derived from."""
    entries = {int(entry["abstraction_level"]): entry for entry in row.get("prompt_levels") or []}
    for level_id in sorted(entries):
        entry = entries[level_id]
        source = entry.get("derived_from")
        parent = None if source == GRAPH_SOURCE or source is None else (entries.get(int(source)) or {}).get("text")
        yield entry, parent
