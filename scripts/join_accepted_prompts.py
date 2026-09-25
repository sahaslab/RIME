"""Stage 5b: keep the prompt rows the verdict report accepted.

Joins the prompt artifact from scripts/generate_ground_truth_prompts.py to the
flat verdict report from `scripts/filter_ground_truth_prompts.py --report` on
`(clip_id, plan_id)`, drops every row whose disposition is not kept, and writes
the survivors in the prompt schema so that

    scripts/generate_agent_input.py --training --plan-prompts <this output>

reads them unchanged. Training mode needs only `prompt_variants` and
`input_audio` off each row and copies the whole row into `metadata`, so the
output is the input rows verbatim: nothing is renamed, nothing is reordered,
and the fields the later stages read (`prompt_levels`, `metadata`) survive.

`input_audio` is carried through as a string and never opened. Training mode
pairs a prompt with no rendered audio -- `ground_truth_edit_audio` comes out ""
-- so this join does not need the corpus mounted, only the two JSONL files.

Why this is not part of stage 5: the report is deliberately a sidecar, so the
prompt artifact downstream consumers read stays untouched while verdicts are
regenerated. This is the join back, kept separate for the same reason
`--generate-full-manifest` is opt-in.

The prompt artifact runs to hundreds of megabytes, so it is streamed rather
than loaded: only the report -- a fifth the size, and further cut to the kept
dispositions -- is held in memory.

Integrity, because the real failure mode here is joining two files from
different runs rather than a bad row:

  - `prompt_variants` must agree between the two sides of every joined pair.
    They are the same strings written twice, so disagreement means the report
    describes a different generation than the prompts do. Fatal unless
    --allow-variant-mismatch.
  - a prompt row with no report row cannot be shown to have been accepted, so
    it is dropped and counted, never passed through.
  - a kept report row that no prompt row matched is counted and logged; it is
    the other half of the same signal.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from collections import Counter
from collections.abc import Iterator, Mapping
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # The bar is cosmetic, and this stage is pure I/O over two
    tqdm = None      # JSONL files -- it should run in any env, not just the one
                     # carrying the judge's dependencies.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ground_truth.io_utils import load_records, write_jsonl, write_json  # noqa: E402


# Identity is carried by the prompt row already, and `prompt_variants` is the
# same list on both sides -- checked per row, so dropping the copy loses
# nothing and keeps the attached verdict to the columns the report adds.
VERDICT_DUPLICATE_KEYS = ("clip_id", "plan_id", "prompt_variants")

# Keys scripts/generate_agent_input.py --training reads off a row directly. A
# row missing either fails there with a KeyError after the join has already
# been paid for, so it fails here instead.
REQUIRED_AGENT_INPUT_KEYS = ("prompt_variants", "input_audio")


def log_event(message: str) -> None:
    print("[join] %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts-path", type=Path, required=True, help="Prompt JSONL from scripts/generate_ground_truth_prompts.py.")
    parser.add_argument("--report-path", type=Path, required=True, help="Flat verdict JSONL from scripts/filter_ground_truth_prompts.py --report.")
    parser.add_argument("--output-path", type=Path, required=True, help="Where to write the accepted prompt rows.")
    parser.add_argument("--disposition", type=str, default="accept", help="Comma-separated dispositions to keep. Default: %(default)s")
    parser.add_argument("--no-verdict", action="store_true", help="Do not attach the report columns under `rejection`; emit the prompt rows alone.")
    parser.add_argument("--allow-variant-mismatch", action="store_true", help="Warn instead of failing when the two sides disagree on prompt_variants.")
    parser.add_argument("--summary-path", type=Path, default=None, help="Also write the run counts here as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the progress bar.")
    return parser.parse_args()


def index_report(path: Path, keep: frozenset[str]) -> tuple[dict[tuple[str, str], dict[str, Any]], Counter]:
    """Kept report rows by `(clip_id, plan_id)`, and the disposition counts.

    Rows outside `keep` are counted and discarded rather than indexed: on a
    25k-row report two thirds of the file never needs to be held. Uniqueness is
    still checked over every row, kept or not -- a duplicated reject means the
    report is malformed just as much as a duplicated accept does.
    """
    index: dict[tuple[str, str], dict[str, Any]] = {}
    dispositions: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    for row in load_records(path):
        key = (str(row.get("clip_id")), str(row.get("plan_id")))
        if key in seen:
            raise ValueError("'%s' has two rows for clip_id=%s plan_id=%s; the join key is not unique." % (path, key[0], key[1]))
        seen.add(key)
        dispositions[row.get("disposition")] += 1
        if row.get("disposition") in keep:
            index[key] = row
    return index, dispositions


def stream_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Prompt rows one at a time, so the artifact is never resident in full."""
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("'%s' line %d is not JSON: %s" % (path, number, error)) from error


def joined_row(record: Mapping[str, Any], verdict: Mapping[str, Any], attach: bool) -> dict[str, Any]:
    """The prompt row, optionally with the verdict under a single key.

    Follows `ground_truth.rejection.full_manifest_row`: the input keys are
    copied through untouched and the verdict lands under `rejection`, so a
    consumer of the prompt schema keeps working and nothing is renamed.
    """
    row = dict(record)
    if attach:
        row["rejection"] = {key: value for key, value in verdict.items() if key not in VERDICT_DUPLICATE_KEYS}
    return row


def accepted_rows(
    prompts_path: Path,
    report_path: Path,
    index: Mapping[tuple[str, str], Mapping[str, Any]],
    counts: Counter,
    matched: set[tuple[str, str]],
    attach: bool,
    allow_variant_mismatch: bool,
    quiet: bool
) -> Iterator[dict[str, Any]]:
    """Every prompt row whose report row was kept, in input order.

    A generator so `write_jsonl` drains it a row at a time; `counts` and
    `matched` are filled as a side effect and are complete once it is consumed.
    """
    stream = stream_rows(prompts_path)
    if not quiet and tqdm is not None:
        stream = tqdm(stream, desc="joining", unit=" rows")
    for record in stream:
        counts["prompt_rows"] += 1
        key = (str(record.get("clip_id")), str(record.get("plan_id")))
        if record.get("clip_id") is None or record.get("plan_id") is None:
            raise ValueError("A row in '%s' has no clip_id/plan_id to join on: %s" % (prompts_path, key))
        verdict = index.get(key)
        if verdict is None:
            # Either the disposition was not kept or the row is absent from the
            # report entirely. The two are distinguished in the summary by
            # comparing against the disposition counts, and both mean the same
            # thing here: not shown to be accepted, so not written.
            counts["dropped"] += 1
            continue
        matched.add(key)
        missing = [name for name in REQUIRED_AGENT_INPUT_KEYS if name not in record]
        if missing:
            raise ValueError(
                "clip_id=%s plan_id=%s is missing the top-level key(s) %s that "
                "scripts/generate_agent_input.py --training reads." % (key[0], key[1], ", ".join(missing))
            )
        if list(record.get("prompt_variants") or []) != list(verdict.get("prompt_variants") or []):
            counts["variant_mismatch"] += 1
            message = (
                "clip_id=%s plan_id=%s has different prompt_variants in '%s' and '%s'. The report was almost "
                "certainly produced from a different prompt artifact than the one being joined." % (key[0], key[1], prompts_path, report_path)
            )
            if not allow_variant_mismatch:
                raise ValueError(message + " Pass --allow-variant-mismatch to write it anyway.")
            log_event("WARNING: " + message)
        counts["kept"] += 1
        yield joined_row(record, verdict, attach)


def main() -> None:
    args = parse_args()
    keep = frozenset(item.strip() for item in args.disposition.split(",") if item.strip())
    if not keep:
        raise ValueError("--disposition named no dispositions to keep.")

    log_event("reading the report from %s" % args.report_path)
    index, dispositions = index_report(args.report_path, keep)
    log_event("report: %d rows, %s" % (sum(dispositions.values()), ", ".join("%s=%d" % (name, count) for name, count in sorted(dispositions.items(), key=lambda item: str(item[0])))))
    log_event("keeping %s: %d rows" % ("/".join(sorted(keep)), len(index)))
    if not index:
        raise ValueError("No report row has a disposition in %s; there is nothing to join." % sorted(keep))

    counts: Counter = Counter()
    matched: set[tuple[str, str]] = set()
    write_jsonl(
        args.output_path,
        accepted_rows(args.prompts_path, args.report_path, index, counts, matched, not args.no_verdict, args.allow_variant_mismatch, args.quiet)
    )

    # Kept report rows no prompt row claimed. Non-zero means the two files do
    # not describe the same set of plans, which is worth saying out loud even
    # though the output is still internally consistent.
    unmatched = len(index) - len(matched)
    summary = {
        "prompts_path": str(args.prompts_path),
        "report_path": str(args.report_path),
        "output_path": str(args.output_path),
        "kept_dispositions": sorted(keep),
        "report_rows": sum(dispositions.values()),
        "report_dispositions": {str(name): count for name, count in dispositions.items()},
        "prompt_rows": counts["prompt_rows"],
        "written": counts["kept"],
        "dropped": counts["dropped"],
        "unmatched_report_rows": unmatched,
        "variant_mismatches": counts["variant_mismatch"],
        "verdict_attached": not args.no_verdict
    }
    log_event("prompts: %d rows, wrote %d, dropped %d" % (counts["prompt_rows"], counts["kept"], counts["dropped"]))
    if unmatched:
        log_event("WARNING: %d kept report row(s) matched no prompt row." % unmatched)
    log_event("wrote %s" % args.output_path)
    if args.summary_path is not None:
        write_json(args.summary_path, summary)
        log_event("wrote the summary to %s" % args.summary_path)


if __name__ == "__main__":
    main()
