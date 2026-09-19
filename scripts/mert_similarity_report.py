"""MERT cosine similarity between each render and its own source audio.

A standalone report over one render manifest. For every row that did not error
it embeds `audio_path` and `output_path`, takes the cosine between them, ranks
that against the whole corpus, and writes a lean row:

    clip_id, plan_id, graph_description, audio_path, output_path,
    similarity, percentile, percentile_bucket, percentile_n

Deliberately narrower than scripts/calculate_ground_truth_similarity.py, which
resolves a reference per row and can prefer the no-FX baseline. This one always
compares the source against the render, as asked -- so the distance includes
whatever Demucs separation cost the clip, not the effects chain alone. That is a
fine thing to measure, it is just a different thing, and the two sets of numbers
should not be pooled.

The model, the embedding cache and the percentile maths are imported from that
script rather than reimplemented, so both reports agree by construction.

Why the percentile is a second pass: a rank describes a whole population, so it
cannot be computed while scoring is still adding to it. It runs automatically at
the end, and `--label-percentiles` re-runs it alone -- which is what a `--resume`
that appended rows needs, since those rows change the population every earlier
label was ranked against.
"""

import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path
from dataclasses import replace
from collections.abc import Mapping, Sequence
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ground_truth.io_utils import load_records  # noqa: E402
from ground_truth.rejection import load_rejection_config  # noqa: E402

DEFAULT_CONFIG_DIR = Path("configs/ground_truth")

# Rows in this state produced no usable audio. Everything else did, whatever it
# is called: the status vocabulary has drifted across manifest generations, so
# this is a blocklist rather than a whitelist.
FAILED_STATUS = "error"

# Written out for every row. Nothing else from the manifest is carried.
REPORT_FIELDS = ("clip_id", "plan_id", "graph_description", "audio_path", "output_path")


def log_event(message: str) -> None:
    print("[mert] %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message), flush=True)


def load_similarity_module() -> Any:
    """The shared MERT machinery, imported from the sibling script by path.

    `scripts/` is not a package, so this is how the filter loads it too.
    """
    path = Path(__file__).resolve().parent / "calculate_ground_truth_similarity.py"
    spec = importlib.util.spec_from_file_location("ground_truth_similarity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scorable_rows(manifest_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Manifest rows that can be scored, reduced to the fields being reported.

    A row is skipped only for a reason that makes scoring impossible: it errored,
    or one of the two files is missing. Each skip is counted and logged rather
    than passed over silently, so a thin report is visibly thin.
    """
    rows: list[dict[str, Any]] = []
    skipped = {"error": 0, "missing_path": 0, "missing_file": 0}
    for row in manifest_rows:
        if row.get("status") == FAILED_STATUS:
            skipped["error"] += 1
            continue
        source, rendered = row.get("audio_path"), row.get("output_path")
        if not source or not rendered:
            skipped["missing_path"] += 1
            continue
        if not Path(source).exists() or not Path(rendered).exists():
            skipped["missing_file"] += 1
            continue
        rows.append({field: row.get(field) for field in REPORT_FIELDS})
    log_event("%d scorable row(s); skipped %d errored, %d with no path, %d with a missing file" % (
        len(rows), skipped["error"], skipped["missing_path"], skipped["missing_file"]
    ))
    return rows


def load_completed(output_path: Path) -> set[tuple[str, str]]:
    """Pairs already scored without error, so --resume skips them."""
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
            if row.get("error") or row.get("similarity") is None:
                continue
            done.add((str(row.get("clip_id")), str(row.get("plan_id"))))
    return done


def score_rows(
    rows: Sequence[Mapping[str, Any]],
    similarity: Any,
    mert: Any,
    output_path: Path,
    resume: bool = False,
    limit: int | None = None
) -> dict[str, int]:
    """Embed and score each row, writing as it goes."""
    done = load_completed(output_path) if resume else set()
    pending = [row for row in rows if (str(row["clip_id"]), str(row["plan_id"])) not in done]
    skipped = len(rows) - len(pending)
    if limit is not None:
        held_back = max(0, len(pending) - limit)
        pending = pending[:limit]
    else:
        held_back = 0
    log_event("%d to score (%d already scored, %d held back by --limit)" % (len(pending), skipped, held_back))

    embedder = similarity.CachedEmbedder(mert)
    counts = {"scored": 0, "failed": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    total = len(pending)

    with output_path.open(mode, encoding="utf-8") as sink:
        for index, row in enumerate(pending, start=1):
            record = dict(row)
            try:
                source = embedder.embedding(Path(row["audio_path"]))
                rendered = embedder.embedding(Path(row["output_path"]))
                record["similarity"] = similarity.cosine_similarity(source, rendered)
            except Exception as error:            # keep going; one bad file should not kill a long job
                counts["failed"] += 1
                message = "%s: %s" % (type(error).__name__, error)
                record["similarity"] = None
                record["error"] = message
                print("[%d/%d] FAILED %s %s -> %s" % (index, total, row["clip_id"], row["plan_id"], message), file=sys.stderr)
            else:
                counts["scored"] += 1
                if index % 100 == 0 or index == total:
                    log_event("[%d/%d] %s %s cos=%.4f (cache %d hit / %d miss)" % (
                        index, total, row["clip_id"], row["plan_id"], record["similarity"], embedder.hits, embedder.misses
                    ))
            sink.write(json.dumps(record, sort_keys=True))
            sink.write("\n")
            sink.flush()

    counts["cache_hits"] = embedder.hits
    counts["cache_misses"] = embedder.misses
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Render manifest JSONL to score.")
    parser.add_argument("--output", type=Path, required=True, help="Output report JSONL.")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR, help="Ground-truth config directory. Default: %(default)s")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Embedding cache directory. Default: from rejection.yaml")
    parser.add_argument("--layer", type=str, default=None, help="'all' to average the layer stack, or 1-12. 12 reproduces fadtk.MERTModel().")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Default: autodetect.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N rows.")
    parser.add_argument("--resume", action="store_true", help="Skip pairs already scored without error.")
    parser.add_argument("--label-percentiles", action="store_true", help="Only relabel an existing report with percentiles, then exit. No model, no GPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    similarity = load_similarity_module()

    mert = load_rejection_config(args.config_dir).mert
    updates: dict[str, Any] = {}
    if args.cache_dir is not None:
        updates["cache_dir"] = args.cache_dir
    if args.layer is not None:
        updates["layer"] = args.layer
    if args.device is not None:
        updates["device"] = args.device
    if updates:
        mert = replace(mert, **updates)

    if args.label_percentiles:
        if not args.output.exists():
            raise SystemExit("Nothing to label: '%s' does not exist." % args.output)
        labels = similarity.label_percentiles(args.output)
        log_event("labelled %d row(s) against a population of %d -> %s" % (labels["labelled"], labels["population"], args.output))
        return

    if not args.manifest.exists():
        raise SystemExit("Manifest '%s' does not exist." % args.manifest)

    manifest_rows = load_records(args.manifest)
    log_event("%d manifest row(s) from %s" % (len(manifest_rows), args.manifest))
    rows = scorable_rows(manifest_rows)
    if not rows:
        raise SystemExit("No scorable rows in '%s'." % args.manifest)

    sources = {row["audio_path"] for row in rows}
    log_event("%d unique source(s) shared across %d row(s), so %d embeddings rather than %d" % (
        len(sources), len(rows), len(sources) + len({row["output_path"] for row in rows}), 2 * len(rows)
    ))

    counts = score_rows(rows, similarity, mert, args.output, resume=args.resume, limit=args.limit)
    log_event("scored %d, failed %d (cache %d hit / %d miss)" % (
        counts["scored"], counts["failed"], counts["cache_hits"], counts["cache_misses"]
    ))

    labels = similarity.label_percentiles(args.output)
    log_event("percentiles over a population of %d -> %s" % (labels["population"], args.output))
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
