"""Stage 5: flag generated ground-truth rows for rejection.

Reads the prompt artifact, raises four flags per row, and writes a verdict
report beside it. The input file is never modified: downstream consumers read
`prompt_variants` and `prompt_levels` out of it and must keep seeing what they
saw before.

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
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
from dotenv import load_dotenv
from litellm import completion
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ground_truth.io_utils import load_records  # noqa: E402
from ground_truth.operators import load_operator_registry  # noqa: E402
from ground_truth.rejection import (  # noqa: E402
    DISPOSITION_REVIEW,
    RejectionConfig,
    build_context,
    parse_verdict,
    index_analysis,
    disposition_for,
    iter_level_jobs,
    similarity_flag,
    triggered_rules,
    full_manifest_row,
    render_retry_note,
    render_row_prompt,
    system_prompt_for,
    reachability_report,
    render_level_prompt,
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
    parser.add_argument("--include-mert", action="store_true", help="Add a MERT original-vs-edited similarity flag, computing any pair not already scored.")
    parser.add_argument("--render-manifest", type=Path, default=None, help="Render manifest for --include-mert. Default: from rejection.yaml")
    parser.add_argument("--similarity-path", type=Path, default=None, help="Precomputed similarity JSONL to join, e.g. the sbatch's OUTPUT_PATH. Default: from rejection.yaml")
    return parser.parse_args()


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
        prompts_path=override(args.prompts_path, config.prompts_path),
        analysis_path=override(args.analysis_path, config.analysis_path),
        output_path=override(args.output_path, config.output_path),
        max_attempts=override(args.max_attempts, config.max_attempts),
        max_workers=override(args.max_workers, config.max_workers),
        criteria_filter=(config.criteria_filter if args.criteria is None else tuple(part.strip() for part in args.criteria.split(",") if part.strip())),
        subsample_num=override(args.subsample_num, config.subsample_num),
        seed=override(args.seed, config.seed)
    )


def preflight_model(model: Any) -> None:
    """Fail on a missing credential before the pool starts, not 4000 rows in."""
    import os
    import importlib.util

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

    Mirrors `call_model` in the prompt generator rather than sharing it: that
    one is bound at import time by name, which scripts/prompt_lab_ui.py depends
    on when it patches `litellm.completion` to record generation runs.
    """
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
    similarity_index: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
) -> dict[str, Any]:
    """Every selected criterion's verdict for one row."""
    clip_id = str(row.get("clip_id"))
    analysis = analysis_index.get(clip_id)
    context = build_context(row, reader, config, analysis)
    flags: dict[str, Any] = {}
    prompts: list[tuple[str, str]] = []

    for criterion in config.selected_criteria():
        if criterion.per == "level":
            flags.update(_evaluate_levels(row, criterion, config, ladder, dry_run, prompts))
            continue
        flags.update(_evaluate_row_criterion(row, criterion, config, context, dry_run, prompts))

    if similarity_index is not None:
        # Joined rather than computed inline: embedding is a GPU job over audio
        # and judging is an I/O-bound job over text, so they are separate passes
        # sharing one artifact.
        key = (clip_id, str(row.get("plan_id")))
        flags["similarity"] = similarity_flag(similarity_index.get(key), config.mert)

    disposition, reasons = disposition_for(flags, config)
    result = {
        "clip_id": clip_id,
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


def _evaluate_levels(row, criterion, config, ladder, dry_run, prompts) -> dict[str, Any]:
    """One call per abstraction level, producing both binary flags."""
    per_level: dict[str, Any] = {}
    per_pair: dict[str, Any] = {}

    for entry, parent_text in iter_level_jobs(row, ladder):
        level_id = int(entry["abstraction_level"])
        # Re-run the mechanical checks rather than trusting the stored result:
        # generation keeps text that still fails after its attempt budget, and
        # the ladder may have moved since. `abstraction_version` is compared by
        # the caller, so a drift is reported rather than judged.
        hard = list(entry.get("violations") or [])
        pair_key = "%s->%s" % (entry.get("derived_from"), level_id)

        if dry_run:
            prompt = render_level_prompt(row, entry, parent_text, criterion, config, hard)
            prompts.append(("%s:level%d" % (criterion.id, level_id), prompt))
            per_level[str(level_id)] = {"verdict": None, "hard_violations": hard, "dry_run": True}
            per_pair[pair_key] = {"verdict": None, "dry_run": True}
            continue

        prompt = render_level_prompt(row, entry, parent_text, criterion, config, hard)
        parsed, problems, attempts = judge(prompt, criterion, config)
        if parsed is None:
            per_level[str(level_id)] = {"verdict": None, "judge_error": list(problems), "attempts": attempts, "hard_violations": hard}
            per_pair[pair_key] = {"verdict": None, "judge_error": list(problems), "attempts": attempts}
            continue
        rules_part = parsed.get("al_rules") or {}
        cons_part = parsed.get("al_consistency") or {}
        # A hard-check failure is a rule failure whatever the judge thinks.
        verdict = "fail" if hard else rules_part.get("verdict")
        per_level[str(level_id)] = {
            "verdict": verdict,
            "judged_verdict": rules_part.get("verdict"),
            "failed_rubric_items": rules_part.get("failed_rubric_items") or [],
            "hard_violations": hard,
            "rationale": parsed.get("rationale"),
            "attempts": attempts
        }
        per_pair[pair_key] = {
            "verdict": cons_part.get("verdict"),
            "violated_constraints": cons_part.get("violated_constraints") or [],
            "rationale": parsed.get("rationale"),
            "attempts": attempts
        }

    return {
        "al_rules": _rollup(per_level, "per_level"),
        "al_consistency": _rollup(per_pair, "per_pair")
    }


def _rollup(parts: Mapping[str, Any], key: str) -> dict[str, Any]:
    """A binary flag from its per-level parts: fail if any part fails."""
    verdicts = [part.get("verdict") for part in parts.values()]
    if any(verdict == "fail" for verdict in verdicts):
        verdict = "fail"
    elif verdicts and all(verdict == "pass" for verdict in verdicts):
        verdict = "pass"
    else:
        verdict = None
    flag: dict[str, Any] = {"verdict": verdict, key: dict(parts)}
    if any(part.get("judge_error") for part in parts.values()):
        flag["judge_error"] = True
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


def build_similarity_index(
    config: RejectionConfig,
    render_manifest: Path | None,
    similarity_path: Path | None = None
) -> dict[tuple[str, str], Mapping[str, Any]]:
    """The similarity rows for --include-mert, computing any that are missing.

    Imported lazily and by path, because the similarity module pulls in torch
    and fadtk and a run without --include-mert should not pay for either.

    Scoring here is a convenience for small runs. For a corpus it is far cheaper
    to run scripts/calculate_ground_truth_similarity.py on a GPU node first, at
    which point every pair is already in `similarity_path` and this only reads
    it back.
    """
    import importlib.util

    module_path = Path(__file__).resolve().parent / "calculate_ground_truth_similarity.py"
    spec = importlib.util.spec_from_file_location("ground_truth_similarity", module_path)
    similarity = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(similarity)

    mert = config.mert
    if similarity_path is not None:
        mert = replace(mert, similarity_path=similarity_path)
    manifest_path = render_manifest or mert.render_manifest
    index = similarity.load_similarity_index(mert.similarity_path)
    if index:
        log_event("read %d similarity rows from %s" % (len(index), mert.similarity_path))

    if manifest_path is None:
        if not index:
            raise ValueError(
                "--include-mert needs a render manifest to score pairs, or an existing %s to read. "
                "Pass --render-manifest or set rejection.mert_similarity.render_manifest." % mert.similarity_path
            )
        return index

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise ValueError("Render manifest '%s' does not exist." % manifest_path)

    manifest_rows = load_records(manifest_path)
    unscored = [
        row for row in manifest_rows
        if (str(row.get("clip_id")), str(row.get("plan_id"))) not in index
    ]
    if unscored:
        log_event("scoring %d unscored pair(s) with %s layer=%s" % (len(unscored), mert.model, mert.layer))
        counts = similarity.compute_similarities(unscored, mert, mert.similarity_path, resume=True)
        log_event("similarity: scored %d, skipped %d, failed %d" % (counts["scored"], counts["skipped"], counts["failed"]))
        index = similarity.load_similarity_index(mert.similarity_path)
    return index


def main() -> None:
    args = parse_args()
    config = resolve_config(args)

    if args.reachability:
        print_reachability(config, args.config_dir)
        return

    registry = load_operator_registry(args.config_dir)
    ladder, lexicon = load_abstraction_config(config_dir=args.config_dir, registry=registry)
    reader = ChainReader(lexicon=lexicon, registry=registry)

    analysis_index: dict[str, Mapping[str, Any]] = {}
    if config.analysis_path and config.analysis_path.exists():
        analysis_index = index_analysis(load_records(config.analysis_path))
        log_event("re-joined %d analysis records from %s" % (len(analysis_index), config.analysis_path))
    elif config.analysis_path:
        log_event("WARNING: analysis manifest %s not found; the stylistic criterion will report no_caption" % config.analysis_path)

    similarity_index: dict[tuple[str, str], Mapping[str, Any]] | None = None
    if args.include_mert:
        similarity_index = build_similarity_index(config, args.render_manifest, args.similarity_path)

    if args.generate_full_manifest and config.full_manifest_path is None:
        raise ValueError("--generate-full-manifest needs paths.full_manifest set in rejection.yaml.")

    rows = load_records(config.prompts_path)
    if config.subsample_num > 0:
        rows = random.Random(config.seed).sample(rows, config.subsample_num)
    if args.limit is not None:
        rows = rows[:args.limit]

    done = load_completed(config.output_path) if args.resume else set()
    pending = [row for row in rows if (str(row.get("clip_id")), str(row.get("plan_id"))) not in done]
    log_event("%d rows, %d already done, %d to evaluate" % (len(rows), len(rows) - len(pending), len(pending)))

    drift = {row.get("abstraction_version") for row in pending} - {ladder.version, None}
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
        for row in pending:
            result = evaluate_row(row, config, reader, ladder, analysis_index, True, similarity_index)
            print("=" * 100)
            print("clip %s  plan %s" % (result["clip_id"], result["plan_id"]))
            for flag, payload in result["flags"].items():
                print("  %-16s %s" % (flag, json.dumps({k: v for k, v in payload.items() if k not in ("per_level", "per_pair")}, sort_keys=True)))
            if dry_manifest is not None:
                # The prompts are a dry-run artifact, not part of the verdict.
                verdict = {key: value for key, value in result.items() if key != "prompts"}
                dry_manifest.write(json.dumps(full_manifest_row(row, verdict), sort_keys=True))
                dry_manifest.write("\n")
            for item in result.get("prompts", []):
                print("-" * 100)
                print("--- PROMPT [%s] ---" % item["criterion"])
                print(item["prompt"])
        if dry_manifest is not None:
            dry_manifest.close()
            log_event("wrote the full manifest to %s" % config.full_manifest_path)
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

    written = 0
    failed = 0
    # A single writer on the main thread. Appending from pool workers would
    # interleave partial lines and corrupt the file resume reads back.
    with config.output_path.open(mode, encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {
                executor.submit(evaluate_row, row, config, reader, ladder, analysis_index, False, similarity_index): row
                for row in pending
            }
            for future in as_completed(futures):
                row = futures[future]
                try:
                    result = future.result()
                except Exception as error:                        # keep going; one bad row should not kill a long job
                    failed += 1
                    result = {
                        "clip_id": str(row.get("clip_id")),
                        "plan_id": row.get("plan_id"),
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
                written += 1
                if written % 50 == 0:
                    log_event("%d/%d written" % (written, len(pending)))

    if review_handle is not None:
        review_handle.close()
    if manifest_handle is not None:
        manifest_handle.close()
        log_event("wrote the full manifest to %s" % config.full_manifest_path)
    log_event("wrote %d rows (%d failed) to %s" % (written, failed, config.output_path))


if __name__ == "__main__":
    main()
