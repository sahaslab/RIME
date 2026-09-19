import argparse
import importlib.util
import json
import os
import random
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv
from litellm import completion
from tqdm import tqdm
from typing import Any

from ground_truth.abstraction import (
    STRUCTURAL_OPERATORS,
    AbstractionLadder,
    AbstractionLevel,
    ChainProfile,
    ChainReader,
    ParamDescriptor,
    load_abstraction_config
)
from ground_truth.io_utils import load_records
from ground_truth.operators import load_operator_registry
from ground_truth.summarization import (
    ModelSettings,
    SummarizationConfig,
    load_summarization_config
)

load_dotenv()

# `--config-dir` is what locates summarization.yaml, so it cannot itself be read
# from that file. Every other default lives there.
DEFAULT_CONFIG_DIR = Path("configs/ground_truth")

# Credential sources litellm's Bedrock path can draw on. Any single one of these
# suffices, so the preflight only complains when none is present.
#
# AWS_BEARER_TOKEN_BEDROCK is a Bedrock API key and is listed first because it
# is the odd one out: litellm sends it as an `Authorization: Bearer` header and
# skips SigV4 entirely, so none of the other variables need to be set alongside
# it. boto3 is still required either way, for botocore's request builder.
AWS_CREDENTIAL_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
)

# Shared, read-only state for the worker pool.
WORKER_LADDER: AbstractionLadder | None = None
WORKER_READER: ChainReader | None = None
WORKER_LEVELS: tuple[AbstractionLevel, ...] = ()
WORKER_MAX_ATTEMPTS = 3
WORKER_MODEL: ModelSettings | None = None
# Set by `main()` for the duration of the run; left None when there is no bar,
# which is how every caller outside `main()` (the prompt lab UI) runs.
WORKER_ON_PROMPT: Callable[[], None] | None = None

# Model settings memoized per config directory, for callers that reach
# `build_prompt_chain` without going through `main()` (the prompt lab UI does).
_MODEL_SETTINGS_CACHE: dict[Path, ModelSettings] = {}


def log_event(message: str) -> None:
    """Timestamped run line, in the same shape as the filter script's.

    Routed through `tqdm.write` rather than `print` because these land while
    the progress bar is live, and a bare write to stdout smears it. With no bar
    open it is an ordinary stdout write.
    """
    tqdm.write("[prompts] %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate iterative prompt rewrites for ground-truth graph descriptions."
    )
    # Every flag below defaults to None so that "not passed" is distinguishable
    # from "passed a value that happens to equal the configured one". Anything
    # left unset falls back to summarization.yaml.
    parser.add_argument(
        "--plans-path",
        type=Path,
        default=None,
        help="Input JSONL of subsampled plans. Default: summarization.paths.plans",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output JSONL path for generated prompt summaries. "
             "Default: summarization.paths.output",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory holding the operators, distributions, abstraction and "
             "summarization configs. Default: %(default)s",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="litellm model id to generate with. Default: summarization.model.name",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default=None,
        help=(
            "Comma-separated abstraction levels to emit. Levels they derive from are "
            "generated regardless. Default: summarization.run.levels, or every level "
            "in the ladder when that is empty."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Generation attempts per level before keeping a text that still fails its "
             "hard checks. Default: summarization.run.max_attempts",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of concurrent graph workers for API-bound generation. "
             "Default: summarization.run.max_workers",
    )
    parser.add_argument(
        "--subsample-num",
        type=int,
        default=None,
        help="How much of the data to randomly sample (for testing). "
             "Default: summarization.run.subsample_num",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for --subsample-num. Default: summarization.run.seed",
    )
    # Not config-backed, unlike every flag above: whether to resume is a fact
    # about one invocation, not a property of the run being reproduced.
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to the output path, skipping plans already written.",
    )
    return parser.parse_args()


def override(flag: Any, configured: Any) -> Any:
    """CLI wins when a flag was passed; otherwise the configured default holds."""
    return configured if flag is None else flag


def default_model_settings(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> ModelSettings:
    """Model settings from `<config_dir>/summarization.yaml`, read once per directory."""
    key = Path(config_dir)
    if key not in _MODEL_SETTINGS_CACHE:
        _MODEL_SETTINGS_CACHE[key] = load_summarization_config(key).model
    return _MODEL_SETTINGS_CACHE[key]


def preflight_model(model: ModelSettings) -> None:
    """Fail early and legibly when a call cannot possibly succeed.

    litellm reports a missing boto3 or absent credentials from deep inside its
    provider stack, so both are checked here instead. The credential test is
    deliberately lenient and only complains when no source at all is visible:
    profiles and instance roles are how this runs on a cluster, and neither sets
    an access-key variable.
    """
    if not model.is_bedrock:
        return
    if importlib.util.find_spec("boto3") is None:
        raise ValueError(
            "Model '%s' uses the Bedrock provider, which requires boto3. "
            "Install it with `pip install boto3`." % model.name
        )

    if model.aws_profile_name:
        return
    if any(os.getenv(name) for name in AWS_CREDENTIAL_ENV_VARS):
        return
    if (Path.home() / ".aws" / "credentials").is_file():
        return
    if (Path.home() / ".aws" / "config").is_file():
        return
    raise ValueError(
        "No AWS credentials are visible for Bedrock model '%s'. Set AWS_BEARER_TOKEN_BEDROCK "
        "to a Bedrock API key, or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or set "
        "AWS_PROFILE, or name a profile under summarization.model.aws_profile_name. "
        "A .env file in the repo root is read." % model.name
    )


def call_model(
    user_prompt: str,
    system_prompt: str,
    max_tokens: int,
    model: ModelSettings,
) -> str:
    """Call the generation model through LiteLLM for one prompt.

    `max_tokens` stays a per-level value from abstraction_levels.yaml; everything
    else about the call comes from summarization.yaml.
    """
    response = completion(
        model=model.name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        **model.completion_kwargs(),
    )
    return response.choices[0].message.content


def render_rules(level: AbstractionLevel) -> str:
    return "\n".join("- %s" % rule for rule in level.rules)


def render_exemplars(level: AbstractionLevel) -> str:
    return "\n".join("- %s" % exemplar for exemplar in level.exemplars)


# Decorations stripped when turning an operator or parameter name into words.
LABEL_PREFIXES = ("apply_", "introduce_")
LABEL_SUFFIXES = ("_effect", "_filter", "_tool")
# Unit suffixes dropped from a parameter label, because the rendered value
# already carries the unit: "cutoff frequency  56 Hz", not "cutoff frequency hz".
UNIT_SUFFIXES = ("_hz", "_db", "_ms", "_seconds")


def readable_label(name: str) -> str:
    """An operator or parameter name as words: `apply_highpass_filter` -> `high pass`."""
    text = name
    for prefix in LABEL_PREFIXES:
        text = text.removeprefix(prefix)
    for suffix in LABEL_SUFFIXES + UNIT_SUFFIXES:
        text = text.removesuffix(suffix)
    return text.replace("_", " ").strip()


def render_chain_summary(profile: ChainProfile) -> str:
    """Describe the chain for a writer, not for a parser.

    Level 0 used to be handed the symbolic graph verbatim, so it was shown
    `separate_audio(audio, description="bass") -> target_stem` and asked to
    describe it. It duly wrote out the function names, the wire names and the
    full-precision floats, which is not what a producer would say. This renders
    the same chain with operator and parameter names as words, values at
    display precision, and no wiring: the stem is named once, and the
    separation and remix that bracket every plan are left out because they are
    how the pipeline applies an edit rather than part of the edit.
    """
    lines: list[str] = []
    stems = [stem for stem in profile.stem_names]
    lines.append("On the %s:" % stems[0] if stems else "On the full mix:")

    by_step: dict[str, list[ParamDescriptor]] = {}
    for descriptor in (*profile.descriptors, *profile.unbanded):
        if descriptor.operator in STRUCTURAL_OPERATORS:
            continue
        by_step.setdefault(descriptor.label, []).append(descriptor)

    for label, descriptors in by_step.items():
        operator = readable_label(descriptors[0].operator)
        if operator == "send return":
            continue
        # A parameter whose name repeats the operator's adds nothing:
        # `apply_gain.gain_db` should read "gain -6.2 dB", not "gain: gain".
        settings = ", ".join(
            d.display if readable_label(d.param) == operator
            else "%s %s" % (readable_label(d.param), d.display)
            for d in descriptors
        )
        lines.append("  %s: %s" % (operator, settings) if settings else "  %s" % operator)

    routing = [d for d in profile.descriptors if d.operator == "send_return"]
    if routing:
        lines.append(
            "  routed to an aux send (%s), rather than placed in line"
            % ", ".join("%s %s" % (readable_label(d.param), d.display) for d in routing)
        )
    return "\n".join(lines)


def render_descriptors(profile: ChainProfile) -> str:
    """Render the band descriptors for a chain as prompt context.

    The values are binned here rather than left to the model, so a level that
    speaks in bands is a deterministic projection of the exact level above it.
    """
    lines: list[str] = []
    for descriptor in profile.descriptors:
        lines.append(
            "  %s.%s -> %s" % (
                descriptor.label,
                descriptor.param,
                ", ".join(descriptor.band_terms)
            )
        )
        if descriptor.hint:
            lines.append("      note: %s" % descriptor.hint)
    return "\n".join(lines)


def render_unbanded(profile: ChainProfile) -> str:
    return "\n".join(
        "  %s.%s" % (descriptor.operator, descriptor.param)
        for descriptor in profile.unbanded
    )


def render_magnitude(profile: ChainProfile) -> str:
    magnitude = profile.magnitude
    if magnitude.value is None:
        return ""
    return "Overall chain intensity: %s (%.2f). Suggested wording: %s." % (
        magnitude.band,
        magnitude.value,
        ", ".join(magnitude.suggested_terms)
    )


def build_level_prompt(
    level: AbstractionLevel,
    source_text: str,
    profile: ChainProfile,
) -> str:
    """Assemble a level's prompt from its declared rules and the chain context.

    Which context blocks appear is driven by the level's `constraints`, so a new
    level needs no code change here.
    """
    constraints = level.constraints
    sections: list[str] = []

    if level.from_graph:
        sections.append(
            "Write the instruction a professional audio producer would give to "
            "produce exactly this chain."
        )
        sections.append("The chain:\n%s" % source_text)
    else:
        sections.append(
            "Rewrite the following music production instruction at a new level of "
            "abstraction, following the rules below."
        )
        sections.append("Instruction to rewrite:\n%s" % source_text)

    sections.append("Rules:\n%s" % render_rules(level))

    if constraints.get("parameter_values") == "banded":
        descriptors = render_descriptors(profile)
        if descriptors:
            sections.append(
                "Parameter descriptors. These are suggestions, not a required "
                "vocabulary: prefer wording that suits the chain as a whole, but keep "
                "each parameter at the magnitude shown.\n%s" % descriptors
            )
        unbanded = render_unbanded(profile)
        if unbanded:
            sections.append(
                "These parameters have no descriptor. Describe them qualitatively, "
                "still without numbers:\n%s" % unbanded
            )

    if profile.overlay_terms and not level.from_graph:
        sections.append(
            "Vocabulary that suits the character of this chain as a whole, to draw "
            "on where it fits: %s." % ", ".join(profile.overlay_terms)
        )

    if constraints.get("focus") == "chain_character":
        magnitude = render_magnitude(profile)
        if magnitude:
            sections.append(magnitude)
        if profile.tags:
            sections.append("Chain character tags: %s" % ", ".join(profile.tags))

    if level.exemplars:
        sections.append("Examples of the target style:\n%s" % render_exemplars(level))

    return "\n\n".join(sections)


def render_retry_note(violations: Sequence[str]) -> str:
    return (
        "\n\nA previous attempt was rejected because it %s. "
        "Write a new instruction that does not." % "; ".join(violations)
    )


def generate_level(
    level: AbstractionLevel,
    source_text: str,
    profile: ChainProfile,
    ladder: AbstractionLadder,
    max_attempts: int,
    model: ModelSettings,
) -> tuple[str, tuple[str, ...], int]:
    prompt = build_level_prompt(level, source_text, profile)
    text = ""
    violations: tuple[str, ...] = ()
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        request = prompt if not violations else prompt + render_retry_note(violations)
        text = (
            call_model(request, ladder.system_prompt_for(level), level.max_tokens, model)
            or ""
        ).strip()
        violations = ladder.check(level, text, profile)
        if not violations:
            break
    return text, violations, attempt


def build_prompt_chain(
    record: Mapping[str, Any],
    levels: Sequence[AbstractionLevel],
    ladder: AbstractionLadder,
    reader: ChainReader,
    max_attempts: int,
    config_dir: Path | str = DEFAULT_CONFIG_DIR,
    model: ModelSettings | None = None,
    on_prompt: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Generate one prompt per level for a single plan.

    `config_dir` and `model` both carry defaults so that a caller which drives
    this directly, rather than through `main()`, does not have to supply them:
    the model then resolves from `config_dir`'s summarization.yaml.

    `on_prompt`, when given, is called once per finished level, so a caller can
    report progress at prompt granularity rather than per plan. It defaults to
    None because the levels of a chain are generated serially here, and the
    only caller that needs a count is the worker pool in `main()`.
    """
    settings = model if model is not None else default_model_settings(config_dir)
    profile = reader.profile(record.get("graph_spec") or [])
    # Not `record["graph_description"]`: that is the symbolic rendering, and
    # handing it to level 0 is what put function names and wire names into the
    # prompts.
    graph_description = render_chain_summary(profile)

    texts: dict[int, str] = {}
    entries: list[dict[str, Any]] = []
    for level in levels:
        source_text = (
            graph_description
            if level.from_graph
            else texts[int(level.derives_from)]
        )
        text, violations, attempts = generate_level(
            level=level,
            source_text=source_text,
            profile=profile,
            ladder=ladder,
            max_attempts=max_attempts,
            model=settings,
        )
        texts[level.id] = text
        entries.append(
            {
                "abstraction_level": level.id,
                "name": level.name,
                "text": text,
                "derived_from": level.derives_from,
                "constraints": dict(level.constraints),
                "rubric": list(level.rubric),
                "attempts": attempts,
                "checks_passed": not violations,
                "violations": list(violations),
            }
        )
        if on_prompt is not None:
            on_prompt()

    return {
        "input_audio": str(record["audio_path"]),
        "clip_id": str(record["clip_id"]),
        "plan_id": str(record["plan_id"]),
        "source": str(record["target_stem"]),
        # Kept as plain strings, ordered by level, for existing consumers.
        "prompt_variants": [entry["text"] for entry in entries],
        "prompt_levels": entries,
        "band_assignments": profile.band_assignments(),
        "chain_magnitude": profile.magnitude.to_dict(),
        "chain_tags": list(profile.tags),
        "abstraction_version": ladder.version,
        "model": settings.name,
        "metadata": record,
    }


class ProgressCounter:
    """Advance a tqdm bar from the worker threads.

    Several workers finish levels at once and tqdm's increment is a
    read-modify-write, so the lock is what keeps the displayed count honest.
    """

    def __init__(self, bar: tqdm) -> None:
        self.bar = bar
        self.lock = Lock()

    def __call__(self) -> None:
        with self.lock:
            self.bar.update(1)


def safe_build_prompt_chain(
    job: tuple[int, Mapping[str, Any]],
) -> tuple[int, dict[str, Any]]:
    if WORKER_LADDER is None or WORKER_READER is None or WORKER_MODEL is None:
        raise RuntimeError("Worker summarization config was not initialized.")
    index, record = job
    result = build_prompt_chain(
        record=record,
        levels=WORKER_LEVELS,
        ladder=WORKER_LADDER,
        reader=WORKER_READER,
        max_attempts=WORKER_MAX_ATTEMPTS,
        model=WORKER_MODEL,
        on_prompt=WORKER_ON_PROMPT,
    )
    return index, result


def row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    """`(clip_id, plan_id)` -- the key a resumed run matches on.

    Reads the same off an input plan and a generated prompt row, since
    `build_prompt_chain` carries both ids through to its output unchanged.
    """
    return str(row.get("clip_id")), str(row.get("plan_id"))


def load_completed(output_path: Path) -> set[tuple[str, str]]:
    """Plans already in the output file, keyed by clip and plan.

    Every line is a finished chain, because a plan whose generation raised is
    logged and skipped rather than written. That is deliberate: an error row in
    a prompt file would sail past the format sniff in
    `filter_ground_truth_prompts.py`, which samples only the first row, and a
    row with no levels to judge is scored as a clean accept. So a failed plan
    is simply absent, and absence is what a resumed run picks up.
    """
    done: set[tuple[str, str]] = set()
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # A run killed mid-write can leave a truncated final line.
                continue
            done.add(row_identity(row))
    return done


def parse_levels(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(part) for part in raw.split(",") if part.strip())


def resolve_config(args: argparse.Namespace) -> SummarizationConfig:
    """Overlay the flags that were actually passed onto summarization.yaml.

    Validation stays in SummarizationConfig, so an override is checked the same
    way a configured value is.
    """
    config = load_summarization_config(args.config_dir)
    model = config.model
    if args.model is not None:
        model = replace(model, name=args.model)
    return replace(
        config,
        model=model,
        plans_path=override(args.plans_path, config.plans_path),
        output_path=override(args.output_path, config.output_path),
        max_attempts=override(args.max_attempts, config.max_attempts),
        max_workers=override(args.max_workers, config.max_workers),
        levels=(
            config.levels if args.levels is None else parse_levels(args.levels)
        ),
        subsample_num=override(args.subsample_num, config.subsample_num),
        seed=override(args.seed, config.seed)
    )


def main() -> None:
    global WORKER_LADDER, WORKER_READER, WORKER_LEVELS, WORKER_MAX_ATTEMPTS, WORKER_MODEL
    global WORKER_ON_PROMPT

    args = parse_args()
    config = resolve_config(args)
    if config.max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")
    preflight_model(config.model)

    registry = load_operator_registry(args.config_dir)
    ladder, lexicon = load_abstraction_config(
        config_dir=args.config_dir,
        registry=registry,
    )

    levels = ladder.select(config.levels) if config.levels else ladder.levels()

    WORKER_LADDER = ladder
    WORKER_READER = ChainReader(lexicon=lexicon, registry=registry)
    WORKER_LEVELS = levels
    WORKER_MAX_ATTEMPTS = config.max_attempts
    WORKER_MODEL = config.model

    data = load_records(config.plans_path)

    if config.subsample_num > 0:
        data = random.Random(config.seed).sample(data, config.subsample_num)

    done = load_completed(config.output_path) if args.resume else set()
    jobs = [
        (index, row)
        for index, row in enumerate(data)
        if row_identity(row) not in done
    ]
    if args.resume:
        log_event(
            "%d plan(s), %d already done, %d to generate" % (
                len(data),
                len(data) - len(jobs),
                len(jobs)
            )
        )

    # Appending is what makes a resumed run additive. A run without --resume
    # truncates instead, so a plain rerun is not silently concatenated onto the
    # rows of the previous one.
    mode = "a" if args.resume and config.output_path.exists() else "w"
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    failed = 0
    # One prompt per level per plan. Counting finished levels rather than
    # finished plans keeps the bar moving while a long chain is still running,
    # and is independent of the order results arrive in. A plan that raises
    # part-way through its chain leaves the rest of its levels uncounted, so the
    # bar can stop short of its total; the tally logged afterwards says why.
    with tqdm(
        total=len(jobs) * len(levels),
        desc="Generating prompts",
        unit="prompt",
    ) as bar:
        WORKER_ON_PROMPT = ProgressCounter(bar)
        # A single writer on the main thread. Appending from the pool workers
        # would interleave partial lines and corrupt the file resume reads back.
        with config.output_path.open(mode, encoding="utf-8") as sink:
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                futures = {
                    executor.submit(safe_build_prompt_chain, job): job
                    for job in jobs
                }
                # Each chain is written as it lands rather than held until the
                # pool drains. Buffering meant a run that died late lost every
                # plan it had already finished, and one unhandled error took the
                # whole job with it -- both of which get likelier as
                # max_workers goes up. The cost is that rows are in completion
                # order, not input order; each carries its own clip and plan id.
                try:
                    for future in as_completed(futures):
                        _, record = futures[future]
                        try:
                            _, result = future.result()
                        except Exception as error:  # one bad plan should not end the run
                            failed += 1
                            clip_id, plan_id = row_identity(record)
                            log_event(
                                "FAILED clip %s plan %s -> %s: %s" % (
                                    clip_id,
                                    plan_id,
                                    type(error).__name__,
                                    error
                                )
                            )
                            continue
                        sink.write(json.dumps(result, sort_keys=True))
                        sink.write("\n")
                        sink.flush()
                except KeyboardInterrupt:
                    # The pool's own shutdown waits without cancelling, so
                    # without this a Ctrl-C would sit through every plan still
                    # queued before the interrupt was allowed through. Rows
                    # already written are flushed to disk, so dropping the
                    # backlog costs only the chains in flight right now.
                    log_event("interrupted -- rerun with --resume to continue")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    log_event(
        "wrote %d plan(s) to %s, %d failed" % (
            len(jobs) - failed,
            config.output_path,
            failed
        )
    )
    if failed:
        log_event("rerun with --resume to retry the %d failed plan(s)." % failed)


if __name__ == "__main__":
    main()
