import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping, Sequence
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion
from typing import Any

from ground_truth.abstraction import (
    AbstractionLadder,
    AbstractionLevel,
    ChainProfile,
    ChainReader,
    load_abstraction_config
)
from ground_truth.io_utils import load_records, write_jsonl
from ground_truth.operators import load_operator_registry

load_dotenv()

MODEL_NAME = "gemini/gemini-3.1-flash-lite-preview"
DEFAULT_PLAN_PATH = Path("derived/ground_truth/subsampled_plans.jsonl")
DEFAULT_OUTPUT_PATH = Path("derived/ground_truth/subsampled_prompts.jsonl")
DEFAULT_CONFIG_DIR = Path("configs/ground_truth")
MAX_ATTEMPTS = 3

# Shared, read-only state for the worker pool.
WORKER_LADDER: AbstractionLadder | None = None
WORKER_READER: ChainReader | None = None
WORKER_LEVELS: tuple[AbstractionLevel, ...] = ()
WORKER_MAX_ATTEMPTS = MAX_ATTEMPTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate iterative prompt rewrites for ground-truth graph descriptions."
    )
    parser.add_argument(
        "--plans-path",
        type=Path,
        default=DEFAULT_PLAN_PATH,
        help="Input JSONL of subsampled plans. Default: %(default)s",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path for generated prompt summaries. Default: %(default)s",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Directory holding operators/distributions/abstraction configs. Default: %(default)s",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="",
        help=(
            "Comma-separated abstraction levels to emit. Levels they derive from are "
            "generated regardless. Default: every level in the ladder."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=MAX_ATTEMPTS,
        help="Generation attempts per level before keeping a text that still fails its "
             "hard checks. Default: %(default)s",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of concurrent graph workers for API-bound generation. Default: %(default)s",
    )
    parser.add_argument(
        "--subsample-num",
        type=int,
        default=-1,
        help="How much of the data to randomly sample (for testing). Default: %(default)s",
    )
    return parser.parse_args()


def call_model(
    user_prompt: str,
    system_prompt: str,
    max_tokens: int,
) -> str:
    """Call the generation model through LiteLLM for one prompt."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError(
            "GEMINI_API_KEY is not set. Please check your .env file or environment variables."
        )

    response = completion(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def render_rules(level: AbstractionLevel) -> str:
    return "\n".join("- %s" % rule for rule in level.rules)


def render_exemplars(level: AbstractionLevel) -> str:
    return "\n".join("- %s" % exemplar for exemplar in level.exemplars)


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
            "recreate the following edit graph exactly."
        )
        sections.append("Edit graph:\n%s" % source_text)
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
) -> tuple[str, tuple[str, ...], int]:
    prompt = build_level_prompt(level, source_text, profile)
    text = ""
    violations: tuple[str, ...] = ()
    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        request = prompt if not violations else prompt + render_retry_note(violations)
        text = (
            call_model(request, ladder.system_prompt_for(level), level.max_tokens) or ""
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
) -> dict[str, Any]:
    profile = reader.profile(record.get("graph_spec") or [])
    graph_description = str(record.get("graph_description", ""))

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
        "metadata": record,
    }


def safe_build_prompt_chain(
    job: tuple[int, Mapping[str, Any]],
) -> tuple[int, dict[str, Any]]:
    if WORKER_LADDER is None or WORKER_READER is None:
        raise RuntimeError("Worker abstraction config was not initialized.")
    index, record = job
    result = build_prompt_chain(
        record=record,
        levels=WORKER_LEVELS,
        ladder=WORKER_LADDER,
        reader=WORKER_READER,
        max_attempts=WORKER_MAX_ATTEMPTS,
    )
    return index, result


def parse_levels(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(part) for part in raw.split(",") if part.strip())


def main() -> None:
    global WORKER_LADDER, WORKER_READER, WORKER_LEVELS, WORKER_MAX_ATTEMPTS

    args = parse_args()
    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1.")

    registry = load_operator_registry(args.config_dir)
    ladder, lexicon = load_abstraction_config(
        config_dir=args.config_dir,
        registry=registry,
    )

    requested = parse_levels(args.levels)
    levels = ladder.select(requested) if requested else ladder.levels()

    WORKER_LADDER = ladder
    WORKER_READER = ChainReader(lexicon=lexicon, registry=registry)
    WORKER_LEVELS = levels
    WORKER_MAX_ATTEMPTS = args.max_attempts

    data = load_records(args.plans_path)

    if args.subsample_num > 0:
        data = random.sample(data, args.subsample_num)

    jobs = [(index, row) for index, row in enumerate(data)]

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        ordered_results = sorted(
            executor.map(safe_build_prompt_chain, jobs),
            key=lambda item: item[0],
        )

    write_jsonl(args.output_path, (result for _, result in ordered_results))


if __name__ == "__main__":
    main()
