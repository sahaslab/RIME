"""Stage 5: flag generated ground-truth rows for rejection.

Reads the prompt artifact, raises four flags per row, and writes a verdict
report beside it. The input file is never modified: downstream consumers read
`prompt_variants` and `prompt_levels` out of it and must keep seeing what they
saw before.

Symbolic criteria only. The embedding-space similarity between the original and
the edited audio lives in scripts/mert_similarity_report.py and
scripts/calculate_ground_truth_similarity.py, and is deliberately not joined in
here: it is a GPU job over audio with an entirely different failure mode and
runtime, while this is an I/O-bound job over text. Keeping them apart also keeps
torch and fadtk off this script's import path. Join the two reports on
`(clip_id, plan_id)` downstream if you want them in one table.

Three modes cost nothing and are the test surface for everything else, since
the repo carries no test suite:

  --reachability   derive which rules can fire at all, and print why not
  --dry-run        assemble and print every prompt, calling no model
  --limit N        do the first N rows only

Resume semantics follow scripts/tag_instruments.py: a row already written with
no error is skipped, a row that failed is retried.
"""

import sys
import json
import time
import random
import argparse
from pathlib import Path
from dataclasses import replace
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from dotenv import load_dotenv
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ground_truth.io_utils import load_records  # noqa: E402
from ground_truth.operators import load_operator_registry  # noqa: E402
from ground_truth.rejection import (  # noqa: E402
    VERDICT_CLEAN,
    VERDICT_LEAKED,
    DISPOSITION_REVIEW,
    RejectionConfig,
    report_row,
    build_context,
    load_captions,
    normalize_row,
    parse_verdict,
    index_analysis,
    is_agent_input,
    disposition_for,
    iter_level_jobs,
    triggered_rules,
    leak_identifiers,
    full_manifest_row,
    render_retry_note,
    render_row_prompt,
    system_prompt_for,
    reachability_report,
    render_level_prompt,
    implementation_leaks,
    load_rejection_config,
)
from ground_truth.abstraction import ChainReader, load_abstraction_config  # noqa: E402

load_dotenv()

DEFAULT_CONFIG_DIR = Path("configs/ground_truth")

# Reused verbatim from scripts/generate_ground_truth_prompts.py: the credential
# sources litellm will find on its own.
AWS_CREDENTIAL_ENV_VARS = (
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_PROFILE",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
)


def log_event(message: str) -> None:
    print("[filter] %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Every flag defaults to None so "not passed" stays distinguishable from
    # "passed the configured value", which is what `override` relies on.
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR, help="Ground-truth config directory. Default: %(default)s")
    parser.add_argument("--prompts-path", type=Path, default=None, help="Input prompt JSONL. Default: from rejection.yaml")
    parser.add_argument("--analysis-path", type=Path, default=None, help="Analysis manifest to re-join for captions and tempo. Default: from rejection.yaml")
    parser.add_argument("--captions-path", type=Path, default=None, help="MusicCaps captions CSV (ytid,caption). Default: from rejection.yaml")
    parser.add_argument("--output-path", type=Path, default=None, help="Output verdict JSONL. Default: from rejection.yaml")
    parser.add_argument("--model", type=str, default=None, help="Override the judge model id.")
    parser.add_argument("--criteria", type=str, default=None, help="Comma-separated criteria to evaluate. Default: all.")
    parser.add_argument("--max-attempts", type=int, default=None, help="Attempts per judge call before recording a parse failure.")
    parser.add_argument("--max-workers", type=int, default=None, help="Thread pool size for model calls.")
    parser.add_argument("--subsample-num", type=int, default=None, help="Random subsample size for smoke tests; -1 disables.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for that subsample.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already written without error.")
    parser.add_argument("--dry-run", action="store_true", help="Print assembled prompts and trigger verdicts; call no model.")
    parser.add_argument("--reachability", action="store_true", help="Print which rules can fire under the current configs, then exit.")
    parser.add_argument("--generate-full-manifest", action="store_true", help="Also write every input record in full with its verdict attached, to paths.full_manifest.")
    parser.add_argument("--report-path", type=Path, default=None, help="Write the flat per-flag report here: prompt_variants, ids, one column per flag. Default: paths.report")
    parser.add_argument("--report", action="store_true", help="Write the flat per-flag report to paths.report.")
    parser.add_argument(
        "--omission-reject",
        action="store_true",
        help="Treat a consistency omission as disqualifying. Off by default: at high abstraction a dropped processor is often terseness, not a falsehood."
    )
    parser.add_argument(
        "--include_al_0",
        action="store_true",
        help="Also judge abstraction level 0. Off by default: AL0 transcribes the graph, so its rubric largely duplicates the mechanical checks."
    )
    parser.add_argument(
        "--without-captions",
        action="store_true",
        help="Drop caption-dependent criteria, for a corpus with no captions. Otherwise they report no_caption, which sends every row to review."
    )
    return parser.parse_args()


def _with_omission(reject_failures: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """`reject_failures` with omission added for the consistency flag.

    Additive rather than replacing the configured list, so `--omission-reject`
    means "and omission too" whatever the config already rejects.
    """
    updated = {name: tuple(values) for name, values in reject_failures.items()}
    current = updated.get("al_consistency", ())
    if "omission" not in current:
        updated["al_consistency"] = current + ("omission",)
    return updated


def override(flag: Any, configured: Any) -> Any:
    """CLI wins when a flag was passed; otherwise the configured default holds."""
    return configured if flag is None else flag


def resolve_config(args: argparse.Namespace) -> RejectionConfig:
    """Overlay the flags that were actually passed onto rejection.yaml.

    Validation stays in RejectionConfig, so an override is checked the same way
    a configured value is.
    """
    config = load_rejection_config(args.config_dir)
    model = config.model
    if args.model is not None:
        model = replace(model, name=args.model)
    return replace(
        config,
        model=model,
        skip_requires=("caption",) if args.without_captions else config.skip_requires,
        include_al0=args.include_al_0 or config.include_al0,
        reject_failures=_with_omission(config.reject_failures) if args.omission_reject else config.reject_failures,
        prompts_path=override(args.prompts_path, config.prompts_path),
        analysis_path=override(args.analysis_path, config.analysis_path),
        captions_path=override(args.captions_path, config.captions_path),
        output_path=override(args.output_path, config.output_path),
        max_attempts=override(args.max_attempts, config.max_attempts),
        max_workers=override(args.max_workers, config.max_workers),
        criteria_filter=(config.criteria_filter if args.criteria is None else tuple(part.strip() for part in args.criteria.split(",") if part.strip())),
        report_path=override(args.report_path, config.report_path),
        subsample_num=override(args.subsample_num, config.subsample_num),
        seed=override(args.seed, config.seed)
    )


def preflight_model(model: Any) -> None:
    """Fail on a missing credential before the pool starts, not 4000 rows in."""
    import os
    import importlib.util

    # Warm litellm here rather than letting a pool worker pay for it. The
    # import costs ~50s on this filesystem, and inside a thread it looks like
    # the run has hung before the first verdict appears. Doing it in preflight
    # makes that time visible and attributable.
    log_event("importing litellm (slow on this filesystem, once per run)")
    import litellm  # noqa: F401

    if not model.is_bedrock:
        return
    if importlib.util.find_spec("boto3") is None:
        raise ValueError("Judge model '%s' targets Bedrock but boto3 is not installed." % model.name)
    if model.aws_profile_name:
        return
    if any(os.environ.get(name) for name in AWS_CREDENTIAL_ENV_VARS):
        return
    home = Path.home()
    if (home / ".aws" / "credentials").exists() or (home / ".aws" / "config").exists():
        return
    raise ValueError(
        "Judge model '%s' targets Bedrock but no credentials were found. Set one of %s, "
        "give rejection.yaml an aws_profile_name, or write ~/.aws/credentials. A .env file "
        "in the repo root is read." % (model.name, ", ".join(AWS_CREDENTIAL_ENV_VARS))
    )


def call_judge(user_prompt: str, system_prompt: str, max_tokens: int, model: Any) -> str:
    """One judge call through litellm.

    litellm is imported here, not at module scope, because importing it takes
    about 50 seconds on this filesystem and the three modes that call no model
    -- `--dry-run`, `--reachability` and `--help` -- were all paying it before
    doing any work. Same reason `tag_instruments.py` imports torch inside its
    tagger rather than at the top.

    Deliberately not shared with `call_model` in the prompt generator: that one
    is bound at import time by name, which scripts/prompt_lab_ui.py depends on
    when it patches `litellm.completion` to record generation runs. The prompt
    lab only ever loads that script, so this one is free to import late.
    """
    from litellm import completion

    response = completion(
        model=model.name,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=max_tokens,
        **model.completion_kwargs()
    )
    return response.choices[0].message.content


def judge(prompt: str, criterion: Any, config: RejectionConfig) -> tuple[dict[str, Any] | None, tuple[str, ...], int]:
    """A parsed verdict, with a bounded retry on an unusable response.

    The generator keeps temperature above zero so a re-send differs; the judge
    runs deterministic and relies on the retry note changing the input instead.
    """
    system_prompt = system_prompt_for(criterion, config)
    problems: tuple[str, ...] = ()
    parsed: dict[str, Any] | None = None
    attempt = 0
    while attempt < config.max_attempts:
        attempt += 1
        request = prompt if not problems else prompt + render_retry_note(problems)
        text = call_judge(request, system_prompt, criterion.max_tokens, config.model) or ""
        parsed, problems = parse_verdict(text, criterion)
        if parsed is not None:
            break
    return parsed, problems, attempt


def evaluate_row(
    row: Mapping[str, Any],
    config: RejectionConfig,
    reader: ChainReader,
    ladder: Any,
    analysis_index: Mapping[str, Mapping[str, Any]],
    dry_run: bool,
    caption_index: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Every selected criterion's verdict for one row.

    Accepts either input format; agent input is unwrapped so every criterion
    below sees the same prompt-row shape.
    """
    row, _audio = normalize_row(row)
    clip_id = str(row.get("clip_id"))
    analysis = analysis_index.get(clip_id)
    # Built once and shared: the mechanical re-run and the trigger context both
    # need it, and rebuilding it per level would walk the graph three times.
    graph_spec = (row.get("metadata") or {}).get("graph_spec") or []
    profile = reader.profile(graph_spec)
    # Per row, because the step labels and stem handles come from this graph.
    identifiers = sorted(leak_identifiers(reader.registry, graph_spec) - set(config.leak_allow))
    caption = (caption_index or {}).get(clip_id)
    context = build_context(row, reader, config, analysis, caption)
    flags: dict[str, Any] = {}
    prompts: list[tuple[str, str]] = []

    for criterion in config.selected_criteria():
        if criterion.per == "level":
            flags.update(_evaluate_levels(row, criterion, config, ladder, profile, identifiers, dry_run, prompts))
            continue
        flags.update(_evaluate_row_criterion(row, criterion, config, context, dry_run, prompts))

    disposition, reasons = disposition_for(flags, config)
    result = {
        "clip_id": clip_id,
        # Carried so the report shows the caption the judge actually read.
        "caption": context["plan"]["caption"],
        "plan_id": row.get("plan_id"),
        "rejection_version": config.version,
        "judge_model": config.model.name,
        "abstraction_version": row.get("abstraction_version"),
        "flags": flags,
        "disposition": disposition,
        "reject_reasons": list(reasons)
    }
    if dry_run:
        result["prompts"] = [{"criterion": name, "prompt": text} for name, text in prompts]
    return result


def _evaluate_levels(row, criterion, config, ladder, profile, identifiers, dry_run, prompts) -> dict[str, Any]:
    """One call per abstraction level, producing both binary flags.

    The mechanical checks and the judge's rubric verdict are reported side by
    side and never combined. They answer different questions: the checks are a
    property of the text against the ladder's contract, the rubric is a property
    the checks cannot express. Folding a mechanical failure into `al_rules`
    would have rejected 98.8% of the MusicCaps corpus for reasons about the
    generator and the checker rather than about the plan.

    Every judged level is judged regardless of its mechanical status. A `judged`
    column that were null wherever the checks failed would carry no
    information, and being able to compare the two is the point of keeping them
    apart.

    Which levels get judged is `config.include_al0`: AL1 and AL2 by default,
    plus AL0 when asked. A level that is not judged still has its mechanical
    status and its leakage checked -- those are free, and AL0 is where almost
    all the leakage is -- but it contributes no `judged` verdict, so it is left
    out of `per_level` and `per_pair` entirely rather than entered as a null.
    A null would drag the rolled-up verdict to null and hide a clean AL1/AL2.
    """
    per_level: dict[str, Any] = {}
    per_pair: dict[str, Any] = {}
    per_leak: dict[str, Any] = {}
    # Kept apart from `per_level`, which now holds only the judged levels. The
    # mechanical status is free and covers every level whether it is judged or
    # not; deriving it from `per_level` silently dropped AL0's the moment AL0
    # stopped being judged.
    per_mechanical: dict[str, str] = {}
    levels_by_id = {level.id: level for level in ladder.levels()}

    for entry, parent_text in iter_level_jobs(row, ladder):
        level_id = int(entry["abstraction_level"])
        pair_key = "%s->%s" % (entry.get("derived_from"), level_id)

        # Re-run rather than trust `entry["violations"]`. The stored result was
        # computed by whatever checker existed at generation time, and that
        # checker could not match a full-precision float -- so on any artifact
        # generated before that fix, half the recorded AL0 failures are wrong.
        # Re-running repairs them without regenerating the corpus.
        level = levels_by_id.get(level_id)
        if level is None:
            violations: list[str] = list(entry.get("violations") or [])
        else:
            violations = list(ladder.check(level, entry.get("text") or "", profile))
        mechanical = "fail" if violations else "pass"
        per_mechanical[str(level_id)] = mechanical

        # Deterministic, and separate from the mechanical checks: leaking
        # `separate_audio` is a different defect from omitting a value, and a
        # text can do either without the other.
        leaks = implementation_leaks(entry.get("text") or "", identifiers, config.leak_syntax_patterns)
        per_leak[str(level_id)] = {
            "verdict": VERDICT_LEAKED if leaks else VERDICT_CLEAN,
            "leaks": leaks
        }
        # Not judged: record the free checks above and move on without a
        # verdict. Deliberately after the mechanical and leak work, so skipping
        # AL0 never costs the leakage signal that mostly lives there.
        if level_id == 0 and not config.include_al0:
            continue

        base = {"mechanical": mechanical, "mechanical_violations": violations}
        prompt = render_level_prompt(row, entry, parent_text, criterion, config, violations, level)

        if dry_run:
            prompts.append(("%s:level%d" % (criterion.id, level_id), prompt))
            per_level[str(level_id)] = dict(base, judged=None, dry_run=True)
            per_pair[pair_key] = {"judged": None, "dry_run": True}
            continue

        parsed, problems, attempts = judge(prompt, criterion, config)
        if parsed is None:
            per_level[str(level_id)] = dict(base, judged=None, judge_error=list(problems), attempts=attempts)
            per_pair[pair_key] = {"judged": None, "judge_error": list(problems), "attempts": attempts}
            continue

        rules_part = parsed.get("al_rules") or {}
        cons_part = parsed.get("al_consistency") or {}
        # Each verdict's own rationale, falling back to a shared one for a
        # judge that returned the older single-rationale shape.
        shared = parsed.get("rationale")
        per_level[str(level_id)] = dict(
            base,
            judged=rules_part.get("verdict"),
            failed_rubric_items=rules_part.get("failed_rubric_items") or [],
            rationale=rules_part.get("rationale") or shared,
            attempts=attempts
        )
        per_pair[pair_key] = {
            "judged": cons_part.get("verdict"),
            "failures": list(cons_part.get("failures") or []),
            "rationale": cons_part.get("rationale") or shared,
            "attempts": attempts
        }

    flags = {
        "al_rules": _rollup(per_level, "per_level"),
        "al_consistency": _rollup(per_pair, "per_pair")
    }
    # The mechanical status travels as its own flag so it reaches the report and
    # the manifest, but it carries no `verdict` key -- `disposition_for` only
    # looks at flags that have one, so this can never reject a row.
    flags["mechanical"] = {
        "per_level": dict(per_mechanical),
        "failed_levels": sorted(name for name, status in per_mechanical.items() if status == "fail")
    }
    leaked = sorted(name for name, part in per_leak.items() if part["verdict"] == VERDICT_LEAKED)
    flags["implementation_leak"] = {
        "verdict": VERDICT_LEAKED if leaked else VERDICT_CLEAN,
        "leaked_levels": leaked,
        "per_level": per_leak
    }
    return flags


def _rollup(parts: Mapping[str, Any], key: str) -> dict[str, Any]:
    """A binary flag from its per-level parts: fail if any judged part fails.

    Keyed on `judged` alone. The mechanical status is reported alongside and is
    deliberately not consulted here.
    """
    verdicts = [part.get("judged") for part in parts.values()]
    if any(verdict == "fail" for verdict in verdicts):
        verdict = "fail"
    elif verdicts and all(verdict == "pass" for verdict in verdicts):
        verdict = "pass"
    else:
        verdict = None
    flag: dict[str, Any] = {"verdict": verdict, key: dict(parts)}
    # Surface the failing parts' justifications to the flag, which is where the
    # flat report reads them. Only the failures: a passing level needs no
    # explanation, and joining all three would bury the one that matters.
    # Labelled by part, because "AL1 invents a compressor" is useless without
    # knowing it was AL1.
    whys = [
        "%s: %s" % (name, part["rationale"])
        for name, part in parts.items()
        if part.get("judged") == "fail" and part.get("rationale")
    ]
    if whys:
        flag["rationale"] = " ".join(whys)
    # Union across pairs, order preserved so the report reads consistently.
    # Which pair failed is in `per_pair`; the flag answers "what kinds of
    # wrongness does this row contain", which is what the policy keys on.
    failures: list[str] = []
    for part in parts.values():
        for item in part.get("failures") or []:
            if item not in failures:
                failures.append(item)
    if failures:
        flag["failures"] = failures
    if any(part.get("judge_error") for part in parts.values()):
        flag["judge_error"] = True
    # Carried up so the report can say WHY the verdict is null. Without it a
    # dry run's nulls are indistinguishable from a judge that errored on every
    # level, which is the one thing a null must never be ambiguous about.
    # `all`, not `any`: a flag is only a dry-run null if nothing was judged.
    if parts and all(part.get("dry_run") for part in parts.values()):
        flag["dry_run"] = True
    return flag


def _evaluate_row_criterion(row, criterion, config, context, dry_run, prompts) -> dict[str, Any]:
    """A row-level criterion: gate on triggers first, judge only if any fired."""
    name = criterion.produces[0]
    if "caption" in criterion.requires and not context["derived"].get("has_caption"):
        return {name: {"verdict": "unclear", "reason": "no_caption", "gated": True, "triggered_rules": []}}

    fired = triggered_rules(criterion, context)
    if not fired:
        # No rule is in play, so there is nothing to adjudicate and no call to
        # pay for. This is the path that makes a dormant rule free.
        return {name: {"verdict": criterion.default_verdict, "gated": True, "triggered_rules": []}}

    rule_ids = [rule.id for rule in fired]
    prompt = render_row_prompt(row, criterion, config, context, fired)
    if dry_run:
        prompts.append((criterion.id, prompt))
        return {name: {"verdict": None, "triggered_rules": rule_ids, "dry_run": True}}

    parsed, problems, attempts = judge(prompt, criterion, config)
    if parsed is None:
        return {name: {"verdict": None, "judge_error": list(problems), "attempts": attempts, "triggered_rules": rule_ids}}
    return {
        name: {
            "verdict": parsed.get("verdict"),
            "triggered_rules": rule_ids,
            "cited_rules": parsed.get("cited_rules") or [],
            "caption_evidence": parsed.get("caption_evidence"),
            "rationale": parsed.get("rationale"),
            "attempts": attempts
        }
    }


def load_completed(output_path: Path) -> set[tuple[str, str]]:
    """Rows already written without a judge error, keyed by clip and plan.

    A row whose judge call failed is deliberately not counted as done, so a
    resumed run retries it.
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
                continue
            if row.get("error"):
                continue
            if any(flag.get("judge_error") for flag in (row.get("flags") or {}).values()):
                continue
            done.add((str(row.get("clip_id")), str(row.get("plan_id"))))
    return done


def print_reachability(config: RejectionConfig, config_dir: Path) -> None:
    registry = load_operator_registry(config_dir)
    _, lexicon = load_abstraction_config(config_dir=config_dir, registry=registry)

    def section(name: str, key: str, default: Any) -> Any:
        with (config_dir / name).open("r", encoding="utf-8") as handle:
            return (yaml.safe_load(handle) or {}).get(key, default)

    report = reachability_report(
        config,
        lexicon,
        registry,
        section("distributions.yaml", "distributions", {}),
        section("motifs.yaml", "motifs", {}),
        section("recipes.yaml", "recipes", [])
    )
    width = max(len(item.rule_id) for item in report)
    counts: dict[str, int] = {}
    for item in report:
        counts[item.verdict] = counts.get(item.verdict, 0) + 1
        print("%-*s  %-6s  %-22s  %s" % (width, item.rule_id, item.kind, item.verdict, item.detail))
    print()
    for verdict in sorted(counts):
        print("%-22s %d" % (verdict, counts[verdict]))


# Keys a prompt row must carry at the top level. Their absence is not a
# recoverable condition: every criterion reads through them, so a row without
# them evaluates to a vacuous pass rather than failing.
REQUIRED_PROMPT_KEYS = ("clip_id", "plan_id", "prompt_levels")


def validate_input_rows(rows: Sequence[Mapping[str, Any]], path: Path) -> bool:
    """Check the input is readable and say which of the two formats it is.

    Both are accepted. Agent input is the default and the richer of the two: it
    is the only artifact carrying the prompt AND the location of the rendered
    audio. This stage only needs the prompt, so either format works; agent
    input stays the default because it also drops rows whose render failed.

    Worth checking explicitly because getting it wrong fails silently rather
    than loudly. A row with no recognizable prompt has no levels to judge and no
    graph to trigger on, so every criterion returns its default and the row is
    written as an accept -- a whole run can look clean while evaluating nothing.
    """
    if not rows:
        raise ValueError("No rows in '%s'." % path)
    sample = rows[0]
    if is_agent_input(sample):
        return True
    missing = [key for key in REQUIRED_PROMPT_KEYS if key not in sample]
    if not missing:
        return False
    raise ValueError(
        "'%s' is neither agent input nor a prompt row: it is missing the top-level key(s) %s. Expected the "
        "output of scripts/generate_agent_input.py (the default) or of "
        "scripts/generate_ground_truth_prompts.py." % (path, ", ".join(missing))
    )


def wants_report(args: argparse.Namespace, config: RejectionConfig) -> bool:
    """Whether to write the flat report.

    Either flag turns it on: `--report` uses the configured path, and
    `--report-path` both turns it on and says where, so naming a path never
    silently does nothing.
    """
    if not (args.report or args.report_path is not None):
        return False
    if config.report_path is None:
        raise ValueError("The flat report needs a path: pass --report-path or set paths.report in rejection.yaml.")
    return True


def row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    """`(clip_id, plan_id)` from either input format."""
    prompt_row, _ = normalize_row(row)
    return str(prompt_row.get("clip_id")), str(prompt_row.get("plan_id"))


def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    if args.reachability:
        print_reachability(config, args.config_dir)
        return

    registry = load_operator_registry(args.config_dir)
    ladder, lexicon = load_abstraction_config(config_dir=args.config_dir, registry=registry)
    reader = ChainReader(lexicon=lexicon, registry=registry)

    dropped = config.dropped_criteria()
    if dropped:
        log_event("dropped %s (requires %s, which this run does not supply)" % (
            ", ".join(item.id for item in dropped),
            ", ".join(sorted({name for item in dropped for name in item.requires}))
        ))
    log_event("evaluating %s" % ", ".join(item.id for item in config.selected_criteria()))

    caption_index: dict[str, str] = {}
    if config.captions_path and config.captions_path.exists():
        caption_index = load_captions(config.captions_path)
        log_event("loaded %d caption(s) from %s" % (len(caption_index), config.captions_path))
    elif not config.skip_requires:
        log_event(
            "WARNING: no captions file at %s. The stylistic criterion will fall back to the mined "
            "aspect phrases, which have the production language stripped out." % config.captions_path
        )

    analysis_index: dict[str, Mapping[str, Any]] = {}
    if config.analysis_path and config.analysis_path.exists():
        analysis_index = index_analysis(load_records(config.analysis_path))
        log_event("re-joined %d analysis records from %s" % (len(analysis_index), config.analysis_path))
    elif config.analysis_path and not config.skip_requires:
        # Not necessarily a problem: newer prompt rows carry the whole analysis
        # record at `metadata.analysis`, which `caption_text` reads directly, so
        # the re-join is only needed for older artifacts that flattened it away.
        log_event(
            "NOTE: analysis manifest %s not found. Rows carrying metadata.analysis are unaffected; "
            "older rows without it will report no_caption." % config.analysis_path
        )

    if args.generate_full_manifest and config.full_manifest_path is None:
        raise ValueError("--generate-full-manifest needs paths.full_manifest set in rejection.yaml.")

    rows = load_records(config.prompts_path)
    agent_input = validate_input_rows(rows, config.prompts_path)
    log_event("%s from %s" % ("agent input" if agent_input else "prompt rows", config.prompts_path))

    if config.subsample_num > 0:
        rows = random.Random(config.seed).sample(rows, config.subsample_num)
    if args.limit is not None:
        rows = rows[:args.limit]

    done = load_completed(config.output_path) if args.resume else set()
    pending = [row for row in rows if row_identity(row) not in done]
    log_event("%d rows, %d already done, %d to evaluate" % (len(rows), len(rows) - len(pending), len(pending)))

    drift = {normalize_row(row)[0].get("abstraction_version") for row in pending} - {ladder.version, None}
    if drift:
        raise ValueError(
            "Prompt rows were written against abstraction_version %s but the loaded ladder is version %s. "
            "The mechanical checks are re-run against the current param_bands.yaml, so a config edit since "
            "generation would be reported as a rejection. Regenerate the prompts or check out the matching "
            "config." % (sorted(str(item) for item in drift), ladder.version)
        )

    if not args.dry_run:
        preflight_model(config.model)

    if args.dry_run:
        # The full manifest is written here too, not only on a real run: its
        # shape is settled by the deterministic half, so it should be checkable
        # without spending a judge call on it.
        dry_manifest = None
        if args.generate_full_manifest:
            config.full_manifest_path.parent.mkdir(parents=True, exist_ok=True)
            dry_manifest = config.full_manifest_path.open("w", encoding="utf-8")
        dry_report = None
        if wants_report(args, config):
            config.report_path.parent.mkdir(parents=True, exist_ok=True)
            dry_report = config.report_path.open("w", encoding="utf-8")
        for row in pending:
            result = evaluate_row(row, config, reader, ladder, analysis_index, True, caption_index)
            print("=" * 100)
            print("clip %s  plan %s" % (result["clip_id"], result["plan_id"]))
            for flag, payload in result["flags"].items():
                print("  %-16s %s" % (flag, json.dumps({k: v for k, v in payload.items() if k not in ("per_level", "per_pair")}, sort_keys=True)))
            verdict = {key: value for key, value in result.items() if key != "prompts"}
            if dry_manifest is not None:
                # The prompts are a dry-run artifact, not part of the verdict.
                dry_manifest.write(json.dumps(full_manifest_row(row, verdict), sort_keys=True))
                dry_manifest.write("\n")
            if dry_report is not None:
                prompt_row, _ = normalize_row(row)
                dry_report.write(json.dumps(report_row(prompt_row, verdict, config), sort_keys=True))
                dry_report.write("\n")
            for item in result.get("prompts", []):
                print("-" * 100)
                print("--- PROMPT [%s] ---" % item["criterion"])
                print(item["prompt"])
        if dry_manifest is not None:
            dry_manifest.close()
            log_event("wrote the full manifest to %s" % config.full_manifest_path)
        if dry_report is not None:
            dry_report.close()
            log_event("wrote the flat report to %s" % config.report_path)
        return

    mode = "a" if args.resume and config.output_path.exists() else "w"
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    review_handle = None
    if config.review_path:
        config.review_path.parent.mkdir(parents=True, exist_ok=True)
        review_handle = config.review_path.open(mode, encoding="utf-8")
    manifest_handle = None
    if args.generate_full_manifest:
        config.full_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_handle = config.full_manifest_path.open(mode, encoding="utf-8")
    report_handle = None
    if wants_report(args, config):
        config.report_path.parent.mkdir(parents=True, exist_ok=True)
        report_handle = config.report_path.open(mode, encoding="utf-8")

    written = 0
    failed = 0
    # A single writer on the main thread. Appending from pool workers would
    # interleave partial lines and corrupt the file resume reads back.
    with config.output_path.open(mode, encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(evaluate_row, row, config, reader, ladder, analysis_index, False, caption_index): row
                for row in pending
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as error:                        # keep going; one bad row should not kill a long job
                    failed += 1
                    clip_id, plan_id = row_identity(row)
                    result = {
                        "clip_id": clip_id,
                        "plan_id": plan_id,
                        "error": "%s: %s" % (type(error).__name__, error)
                    }
                    log_event("FAILED clip %s plan %s -> %s" % (result["clip_id"], result["plan_id"], result["error"]))
                sink.write(json.dumps(result, sort_keys=True))
                sink.write("\n")
                sink.flush()
                if review_handle is not None and result.get("disposition") == DISPOSITION_REVIEW:
                    review_handle.write(json.dumps(result, sort_keys=True))
                    review_handle.write("\n")
                    review_handle.flush()
                if manifest_handle is not None:
                    manifest_handle.write(json.dumps(full_manifest_row(row, result), sort_keys=True))
                    manifest_handle.write("\n")
                    manifest_handle.flush()
                if report_handle is not None:
                    prompt_row, _ = normalize_row(row)
                    report_handle.write(json.dumps(report_row(prompt_row, result, config), sort_keys=True))
                    report_handle.write("\n")
                    report_handle.flush()
                written += 1
                if written % 50 == 0:
                    log_event("%d/%d written" % (written, len(pending)))

    if review_handle is not None:
        review_handle.close()
    if manifest_handle is not None:
        manifest_handle.close()
        log_event("wrote the full manifest to %s" % config.full_manifest_path)
    if report_handle is not None:
        report_handle.close()
        log_event("wrote the flat report to %s" % config.report_path)
    log_event("wrote %d rows (%d failed) to %s" % (written, failed, config.output_path))


if __name__ == "__main__":
    main()
