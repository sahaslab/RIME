"""MERT embedding similarity between the original and the edited audio.

Stage 5b of the ground-truth stack, and the one criterion that needs audio
rather than symbols: it is the only check that can catch an edit too subtle to
hear, which no amount of rule-checking will find.

Runnable standalone -- point it at a render manifest and it writes a similarity
JSONL -- and importable, which is how scripts/filter_ground_truth_prompts.py
calls it under --include-mert. Running it standalone first on a GPU node is the
cheaper order for a large corpus: the filter then only reads the result.

It never triggers a render. Rendering means booting the MCP server and Demucs,
so a plan with no rendered audio reports a null similarity with reason
`not_rendered`.

Which "original"
---------------
The render manifest carries three, and they do not measure the same thing:

  baseline_path      the no-FX remix -- separated and remixed with no effects.
                     Cancels separation artifacts, so the distance is the
                     effects chain rather than Demucs. This is the default.
  source_copy_path   a byte copy of the untouched recording. Conflates
                     separation loss with the edit, so it reads as further away
                     even for a plan that barely touches its stem.
  audio_path         the original, referenced outside the output root.

Which one was used is recorded per row as `reference`, because the two are not
poolable.

Poison plans invert the pair. When `poison_graph_spec` is present the manifest's
`baseline_path` is the *degraded input* and `output_path` is the *repaired*
result, so comparing them would measure repair fidelity rather than edit
strength. Those rows compare the source against the output and are marked
`poison_repair`, and their numbers must not be pooled with the rest either.
"""

import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Iterator, Sequence
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ground_truth.io_utils import load_records  # noqa: E402
from ground_truth.rejection import MertSimilarityConfig, normalize_row, load_rejection_config  # noqa: E402

DEFAULT_CONFIG_DIR = Path("configs/ground_truth")

# MERT-v1-95M's input rate. fadtk carries the same number; it is repeated here
# because the resampling happens before any fadtk object exists.
MERT_SAMPLE_RATE = 24000

# fadtk resamples with these exact settings in FrechetAudioDistance.load_audio.
# Matched so a `layer: 12` run is comparable to standard FAD/KAD numbers
# rather than differing by an interpolation filter.
RESAMPLE_KWARGS = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379492
}

# The one status that means no usable audio. Deliberately a blocklist rather
# than a whitelist: the status vocabulary has drifted across manifest
# generations -- the current renderer writes `rendered`, `skipped_existing`,
# `skipped_harmony` and `error`, while the older `generated-audio` manifests
# carry `skipped_existing` for 998 of 1000 rows whose audio is present and
# valid. Whitelisting `rendered` silently discarded that entire artifact.
FAILED_STATUS = "error"

# Quartile labels for the similarity percentile, ascending by similarity.
# Named for what the number means rather than Q1..Q4: a HIGH similarity is a
# SMALL change, so the top quartile is the one at risk of being inaudible --
# which is the whole reason this criterion exists.
PERCENTILE_BUCKETS = (
    (25.0, "most_audible"),
    (50.0, "audible"),
    (75.0, "subtle"),
    (100.01, "least_audible")
)

REASON_NOT_RENDERED = "not_rendered"
REASON_RENDER_ERROR = "render_error"
REASON_NO_REFERENCE = "no_reference_audio"
REASON_MISSING_FILE = "missing_audio_file"


def log_event(message: str) -> None:
    print("[similarity] %s | %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), message), flush=True)


@dataclass(frozen=True)
class SimilarityPair:
    """One original/edited comparison to make."""

    clip_id: str
    plan_id: str
    reference_path: Path
    edited_path: Path
    reference: str
    poison_repair: bool


def resolve_pair(row: Mapping[str, Any], reference_mode: str) -> tuple[SimilarityPair | None, str | None]:
    """The pair to compare for one render-manifest row, or why there isn't one.

    Returns `(pair, reason)` with exactly one of them set.
    """
    clip_id = str(row.get("clip_id"))
    plan_id = str(row.get("plan_id"))
    if row.get("status") == FAILED_STATUS:
        # An errored row can still leave a stale output file from an earlier
        # partial pass, so the status has to be checked before the path.
        return None, REASON_RENDER_ERROR
    edited = row.get("output_path")
    if not edited or not Path(edited).exists():
        return None, REASON_NOT_RENDERED

    # Inferred from `poison_graph_spec`, not from `baseline_kind`: the older
    # manifests predate that key and omit it entirely.
    poison_repair = bool(row.get("poison_graph_spec"))

    baseline = row.get("baseline_path")
    source = row.get("source_copy_path") or row.get("audio_path")

    if poison_repair:
        # `baseline_path` here is the degraded input, so the only meaningful
        # original is the untouched source.
        candidates = [(source, "source")]
    elif reference_mode == "baseline_only":
        candidates = [(baseline, "baseline")]
    elif reference_mode == "source_only":
        candidates = [(source, "source")]
    else:
        candidates = [(baseline, "baseline"), (source, "source")]

    for candidate, kind in candidates:
        if candidate and Path(candidate).exists():
            return SimilarityPair(
                clip_id=clip_id,
                plan_id=plan_id,
                reference_path=Path(candidate),
                edited_path=Path(edited),
                reference=kind,
                poison_repair=poison_repair
            ), None
    return None, REASON_NO_REFERENCE


def iter_pairs(manifest_rows: Sequence[Mapping[str, Any]], reference_mode: str) -> Iterator[tuple[Mapping[str, Any], SimilarityPair | None, str | None]]:
    for row in manifest_rows:
        pair, reason = resolve_pair(row, reference_mode)
        yield row, pair, reason


def iter_agent_pairs(
    rows: Sequence[Mapping[str, Any]],
    reference_mode: str,
    baseline_index: Mapping[tuple[str, str], str] | None = None
) -> Iterator[tuple[Mapping[str, Any], "SimilarityPair | None", str | None]]:
    """Pairs taken straight from agent-input rows.

    Agent input already carries both sides, so no render manifest is needed.
    One is still accepted: it is the only place the no-FX baseline lives, and
    that reference isolates the effects chain where agent input's `input_audio`
    conflates it with Demucs separation loss.
    """
    for row in rows:
        prompt_row, audio = normalize_row(row)
        identity = {
            "clip_id": prompt_row.get("clip_id"),
            "plan_id": prompt_row.get("plan_id"),
            "status": "agent_input"
        }
        if audio is None:
            yield identity, None, REASON_NOT_RENDERED
            continue

        key = (str(identity["clip_id"]), str(identity["plan_id"]))
        reference_path, reference_kind = audio.reference_path, audio.reference
        # A manifest baseline beats agent input's source reference, except on a
        # poison row where the baseline IS the degraded input.
        if baseline_index and reference_mode != "source_only" and not audio.poison_repair:
            baseline = baseline_index.get(key)
            if baseline and Path(baseline).exists():
                reference_path, reference_kind = baseline, "baseline"
        if reference_mode == "baseline_only" and reference_kind != "baseline":
            yield identity, None, REASON_NO_REFERENCE
            continue

        if not Path(reference_path).exists() or not Path(audio.edited_path).exists():
            yield identity, None, REASON_MISSING_FILE
            continue
        yield identity, SimilarityPair(
            clip_id=key[0],
            plan_id=key[1],
            reference_path=Path(reference_path),
            edited_path=Path(audio.edited_path),
            reference=reference_kind,
            poison_repair=audio.poison_repair
        ), None


def baseline_index_from_manifest(manifest_rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], str]:
    """`(clip_id, plan_id)` -> no-FX baseline path, for upgrading the reference."""
    index: dict[tuple[str, str], str] = {}
    for row in manifest_rows:
        if row.get("status") == FAILED_STATUS or not row.get("baseline_path"):
            continue
        # A poison row's baseline is the degraded input, not a clean reference.
        if row.get("poison_graph_spec"):
            continue
        index[(str(row.get("clip_id")), str(row.get("plan_id")))] = str(row["baseline_path"])
    return index


def embedding_cache_path(cache_dir: Path, audio_path: Path, model_key: str) -> Path:
    """Where one file's pooled embedding is cached.

    Keyed on path, mtime and size alongside the model identity, so re-rendering
    a clip invalidates its embedding rather than silently reusing the old one.
    Same construction as `cache_key` in subsample_ground_truth_plans.py.
    """
    try:
        stat = audio_path.stat()
        mtime, size = stat.st_mtime_ns, stat.st_size
    except OSError:
        mtime, size = 0, 0
    payload = json.dumps(
        {"path": str(audio_path.resolve(strict=False)), "mtime": mtime, "size": size, "model": model_key},
        sort_keys=True
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
    return cache_dir / ("%s.npy" % digest)


class PooledMertModel:
    """fadtk's MERT loader with the pooling replaced.

    Subclassing rather than reimplementing keeps the checkpoint, the
    Wav2Vec2FeatureExtractor, the device placement and the six-minute truncation
    exactly as fadtk has them. The only thing overridden is which hidden
    states survive: fadtk selects a single layer (`out = out[self.layer]`),
    and the default here averages the whole stack.

    Setting `layer: 12` makes this behave as an unmodified `fadtk.MERTModel()`,
    which is what lets the two sets of numbers be compared.
    """

    def __init__(self, config: MertSimilarityConfig) -> None:
        # Imported here so --help and --from-cache do not pay for torch.
        import torch
        import fadtk

        self.torch = torch
        self.config = config
        size = config.model.replace("MERT-", "")

        class _Pooled(fadtk.MERTModel):
            def __init__(inner) -> None:
                super().__init__(size=size, layer=config.numeric_layer)
                inner.name = "%s-layer_%s" % (config.model, config.layer)

            def _get_embedding(inner, audio):
                inputs = inner.processor(audio, sampling_rate=inner.sr, return_tensors="pt").to(inner.device)
                with torch.inference_mode():
                    out = inner.model(**inputs, output_hidden_states=True)
                    # [13 layers, frames, 768], as fadtk stacks them.
                    stacked = torch.stack(out.hidden_states).squeeze()
                    if config.pools_all_layers:
                        return stacked.mean(dim=0)
                    return stacked[inner.config_layer]

        _Pooled.config_layer = config.numeric_layer
        self.model = _Pooled()
        if config.device:
            self.model.device = torch.device(config.device)
        self.model.load_model()
        log_event("loaded %s on %s (layer=%s)" % (self.model.name, self.model.device, config.layer))

    def embed(self, audio_path: Path):
        """The pooled 768-vector for one file, as float32 numpy."""
        import numpy as np

        audio = load_audio_mono_24k(audio_path, self.torch)
        frames = self.model._get_embedding(audio)
        # Mean over time last, so the cache holds one vector per file rather
        # than a frame matrix. Cast up from fadtk's float16 economy: a cosine
        # over 768 dims does not need the rounding.
        pooled = frames.mean(dim=0).detach().float().cpu().numpy()
        return np.asarray(pooled, dtype=np.float32)


def read_audio(audio_path: Path) -> tuple[Any, int]:
    """Raw samples and native rate, channels-first, as the repo loads audio.

    `soundfile` rather than `torchaudio.load`: on torch 2.11 that dispatches to
    torchcodec, which needs libnvrtc and therefore fails outright on a CPU node.
    soundfile is also what every existing loader in this repo uses, with the
    same `always_2d` / float32 / channels-first convention.

    librosa is the fallback for formats libsndfile refuses -- some of the
    reference candidates are the original `.mp3`.
    """
    import numpy as np
    import soundfile as sf

    try:
        data, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        return np.ascontiguousarray(data.T), int(sample_rate)
    except Exception:
        import librosa

        data, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
        data = np.atleast_2d(data)
        return np.ascontiguousarray(data), int(sample_rate)


def load_audio_mono_24k(audio_path: Path, torch: Any):
    """One file as mono float32 at MERT's input rate.

    Deliberately in memory. fadtk's own loader writes a converted copy to
    `<parent>/convert/<sr>/` beside the source, which for these render trees
    would mean writing into shared, possibly read-only artifact directories.
    """
    import torchaudio

    data, sample_rate = read_audio(audio_path)
    waveform = torch.from_numpy(data)
    # Mono first, so the resampler runs once rather than per channel. Same
    # downmix fadtk uses.
    waveform = torch.mean(waveform, 0).unsqueeze(0)
    if sample_rate != MERT_SAMPLE_RATE:
        # transforms.Resample is a plain tensor op, so it is unaffected by the
        # torchcodec problem that rules out torchaudio.load.
        resampler = torchaudio.transforms.Resample(sample_rate, MERT_SAMPLE_RATE, **RESAMPLE_KWARGS)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy()


class CachedEmbedder:
    """Pooled embeddings with a disk cache, loading MERT only when it must.

    The cache is what makes the reference side cheap: one clip's baseline is
    shared by every plan targeting that stem, so a corpus with a thousand edited
    renders needs only a few hundred reference embeddings.
    """

    def __init__(self, config: MertSimilarityConfig, allow_model: bool = True) -> None:
        self.config = config
        self.allow_model = allow_model
        self.cache_dir = config.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model: PooledMertModel | None = None
        self.hits = 0
        self.misses = 0

    def _ensure_model(self) -> PooledMertModel:
        if self._model is None:
            if not self.allow_model:
                raise RuntimeError("Embedding is not cached and --from-cache forbids loading MERT.")
            self._model = PooledMertModel(self.config)
        return self._model

    def embedding(self, audio_path: Path):
        import numpy as np

        cache_path = embedding_cache_path(self.cache_dir, audio_path, self.config.model_key)
        if cache_path.exists():
            self.hits += 1
            return np.load(cache_path)
        vector = self._ensure_model().embed(audio_path)
        # Write through a temp name so an interrupted run cannot leave a
        # truncated .npy that later reads as a cache hit. Saved through an open
        # handle because np.save appends ".npy" to any path that lacks it, which
        # would put the file somewhere the rename below cannot find.
        temp_path = cache_path.with_name(cache_path.name + ".tmp")
        with temp_path.open("wb") as handle:
            np.save(handle, vector)
        temp_path.replace(cache_path)
        self.misses += 1
        return vector


def cosine_similarity(first: Any, second: Any) -> float:
    """Cosine between two pooled embeddings, matching the reference metric."""
    import numpy as np

    a = np.asarray(first, dtype=np.float64).ravel()
    b = np.asarray(second, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def similarity_row(
    manifest_row: Mapping[str, Any],
    pair: SimilarityPair | None,
    reason: str | None,
    value: float | None,
    config: MertSimilarityConfig,
    error: str | None = None
) -> dict[str, Any]:
    """One output row. Always carries the raw value, never only a verdict.

    Recording the number unconditionally is what lets a threshold be chosen, or
    changed, without re-running MERT over the corpus.
    """
    row: dict[str, Any] = {
        "clip_id": str(manifest_row.get("clip_id")),
        "plan_id": str(manifest_row.get("plan_id")),
        "similarity": value,
        "model": config.model,
        "layer": config.layer,
        "render_status": manifest_row.get("status")
    }
    if pair is not None:
        row["reference"] = pair.reference
        row["poison_repair"] = pair.poison_repair
        row["reference_path"] = str(pair.reference_path)
        row["edited_path"] = str(pair.edited_path)
    if reason is not None:
        row["reason"] = reason
    if error is not None:
        row["error"] = error
    return row


def percentile_of(sorted_values: Any, value: float) -> float:
    """Where one value sits in the population, 0-100.

    Percent of the population strictly below, plus half of what ties it. That
    midpoint convention is what keeps a run of identical values centred on one
    percentile instead of all landing at the bottom or the top of their run.
    """
    import numpy as np

    below = int(np.searchsorted(sorted_values, value, side="left"))
    at_or_below = int(np.searchsorted(sorted_values, value, side="right"))
    ties = at_or_below - below
    return 100.0 * (below + 0.5 * ties) / len(sorted_values)


def bucket_for(percentile: float) -> str:
    for upper, name in PERCENTILE_BUCKETS:
        if percentile < upper:
            return name
    return PERCENTILE_BUCKETS[-1][1]


def label_percentiles(path: Path) -> dict[str, int]:
    """Add a global percentile and bucket to every scored row in a file.

    A second pass on purpose. A percentile is a statement about the whole
    population, so it cannot be computed while scoring is still adding to that
    population -- and a `--resume` that appends new rows makes every existing
    label stale. Each row therefore records `percentile_n`, the population it
    was ranked against, so a consumer can tell a fresh label from a stale one
    rather than trusting it blindly.

    Rows with no similarity are left with a null percentile. They are not part
    of the population either: ranking against unscored rows would make the
    percentile depend on how much of the corpus had been rendered.
    """
    import numpy as np

    rows = [row for row in load_records(path)]
    values = sorted(float(row["similarity"]) for row in rows if row.get("similarity") is not None)
    counts = {"labelled": 0, "unscored": 0, "population": len(values)}
    if not values:
        return counts

    sorted_values = np.asarray(values, dtype=np.float64)
    for row in rows:
        if row.get("similarity") is None:
            row["percentile"] = None
            row["percentile_bucket"] = None
            row["percentile_n"] = len(values)
            counts["unscored"] += 1
            continue
        percentile = percentile_of(sorted_values, float(row["similarity"]))
        row["percentile"] = round(percentile, 4)
        row["percentile_bucket"] = bucket_for(percentile)
        row["percentile_n"] = len(values)
        counts["labelled"] += 1

    # Rewritten through a temp file so an interrupted relabel cannot leave the
    # similarity artifact half-labelled.
    temp_path = path.with_name(path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    temp_path.replace(path)
    return counts


def load_similarity_index(path: Path) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Similarity rows by `(clip_id, plan_id)`, for the filter to join against."""
    index: dict[tuple[str, str], Mapping[str, Any]] = {}
    if not Path(path).exists():
        return index
    for row in load_records(Path(path)):
        index[(str(row.get("clip_id")), str(row.get("plan_id")))] = row
    return index


def load_completed(output_path: Path) -> set[tuple[str, str]]:
    """Pairs already written without error, so --resume skips them.

    A row that errored is deliberately not counted, so a resumed run retries it.
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
            done.add((str(row.get("clip_id")), str(row.get("plan_id"))))
    return done


def compute_similarities(
    pairs: Any,
    config: MertSimilarityConfig,
    output_path: Path,
    resume: bool = False,
    limit: int | None = None,
    from_cache: bool = False,
    pairs_only: bool = False
) -> dict[str, int]:
    """Write a similarity row per resolved pair. Returns counts by outcome.

    Takes already-resolved `(identity, pair, reason)` triples rather than raw
    rows, so one scoring loop serves both inputs: `iter_agent_pairs` for agent
    input, `iter_pairs` for a render manifest.
    """
    done = load_completed(output_path) if resume else set()
    jobs = [
        (row, pair, reason)
        for row, pair, reason in pairs
        if (str(row.get("clip_id")), str(row.get("plan_id"))) not in done
    ]
    if limit is not None:
        jobs = jobs[:limit]

    counts = {"scored": 0, "skipped": 0, "failed": 0, "cache_hits": 0, "cache_misses": 0}
    embedder = None if pairs_only else CachedEmbedder(config, allow_model=not from_cache)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and output_path.exists() else "w"
    total = len(jobs)

    with output_path.open(mode, encoding="utf-8") as sink:
        for index, (manifest_row, pair, reason) in enumerate(jobs, start=1):
            if pair is None:
                counts["skipped"] += 1
                _write_row(sink, similarity_row(manifest_row, None, reason, None, config))
                continue
            if pairs_only:
                counts["scored"] += 1
                _write_row(sink, similarity_row(manifest_row, pair, None, None, config, error=None))
                continue
            try:
                reference = embedder.embedding(pair.reference_path)
                edited = embedder.embedding(pair.edited_path)
                value = cosine_similarity(reference, edited)
            except Exception as error:                        # keep going; one bad file should not kill a long job
                counts["failed"] += 1
                message = "%s: %s" % (type(error).__name__, error)
                print("[%d/%d] FAILED %s %s -> %s" % (index, total, pair.clip_id, pair.plan_id, message), file=sys.stderr)
                _write_row(sink, similarity_row(manifest_row, pair, None, None, config, error=message))
                continue
            counts["scored"] += 1
            print("[%d/%d] %s %s  %s  cos=%.4f" % (index, total, pair.clip_id, pair.plan_id, pair.reference, value))
            _write_row(sink, similarity_row(manifest_row, pair, None, value, config))

    if embedder is not None:
        counts["cache_hits"] = embedder.hits
        counts["cache_misses"] = embedder.misses
    return counts


def _write_row(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True))
    handle.write("\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR, help="Ground-truth config directory. Default: %(default)s")
    parser.add_argument("--render-manifest", type=Path, default=None, help="Render manifest JSONL. Default: from rejection.yaml")
    parser.add_argument("--output", type=Path, default=None, help="Output similarity JSONL. Default: from rejection.yaml")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Embedding cache directory. Default: from rejection.yaml")
    parser.add_argument("--model", type=str, default=None, help="fadtk MERT size string, e.g. MERT-v1-95M.")
    parser.add_argument("--layer", type=str, default=None, help="'all' to average the layer stack, or 1-12. 12 reproduces fadtk.MERTModel().")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Default: autodetect.")
    parser.add_argument("--reference", type=str, default=None, choices=["baseline_then_source", "baseline_only", "source_only"], help="Which original to compare against.")
    parser.add_argument("--limit", type=int, default=None, help="Score only the first N rows.")
    parser.add_argument("--resume", action="store_true", help="Skip pairs already scored without error.")
    parser.add_argument("--from-cache", action="store_true", help="Recompute similarities from cached embeddings without loading MERT.")
    parser.add_argument("--pairs-only", action="store_true", help="Resolve and report pairs without embedding anything. No model, no GPU.")
    parser.add_argument("--label-percentiles", action="store_true", help="Only relabel an existing similarity file with global percentiles, then exit. No model, no GPU.")
    parser.add_argument("--no-label-percentiles", action="store_true", help="Skip the percentile pass after scoring, e.g. when this run is one shard of a corpus.")
    return parser.parse_args()


def resolve_mert_config(args: argparse.Namespace) -> MertSimilarityConfig:
    """Overlay the flags that were passed onto rejection.yaml's mert block."""
    from dataclasses import replace

    config = load_rejection_config(args.config_dir).mert
    updates: dict[str, Any] = {}
    if args.render_manifest is not None:
        updates["render_manifest"] = args.render_manifest
    if args.output is not None:
        updates["similarity_path"] = args.output
    if args.cache_dir is not None:
        updates["cache_dir"] = args.cache_dir
    if args.model is not None:
        updates["model"] = args.model
    if args.layer is not None:
        updates["layer"] = args.layer
    if args.device is not None:
        updates["device"] = args.device
    if args.reference is not None:
        updates["reference"] = args.reference
    config = replace(config, **updates)
    # Validation lives in the dataclass, so an override is checked the same way
    # a configured value is.
    return MertSimilarityConfig.from_mapping(
        {
            "model": config.model,
            "layer": config.layer,
            "device": config.device,
            "render_manifest": str(config.render_manifest) if config.render_manifest else None,
            "cache_dir": str(config.cache_dir),
            "similarity_path": str(config.similarity_path),
            "reference": config.reference,
            "similarity_threshold": config.similarity_threshold
        },
        "CLI overrides"
    )


def main() -> None:
    args = parse_args()
    config = resolve_mert_config(args)

    if args.label_percentiles:
        if not config.similarity_path.exists():
            raise SystemExit("Nothing to label: '%s' does not exist." % config.similarity_path)
        counts = label_percentiles(config.similarity_path)
        log_event("labelled %d row(s) against a population of %d, %d unscored -> %s" % (
            counts["labelled"], counts["population"], counts["unscored"], config.similarity_path
        ))
        return

    if config.render_manifest is None:
        raise SystemExit("No render manifest. Pass --render-manifest or set rejection.mert_similarity.render_manifest.")
    if not config.render_manifest.exists():
        raise SystemExit("Render manifest '%s' does not exist." % config.render_manifest)

    manifest_rows = load_records(config.render_manifest)
    log_event("%d manifest rows from %s" % (len(manifest_rows), config.render_manifest))

    counts = compute_similarities(
        iter_pairs(manifest_rows, config.reference),
        config,
        config.similarity_path,
        resume=args.resume,
        limit=args.limit,
        from_cache=args.from_cache,
        pairs_only=args.pairs_only
    )
    log_event(
        "scored %d, skipped %d, failed %d (cache %d hit / %d miss) -> %s" % (
            counts["scored"], counts["skipped"], counts["failed"],
            counts["cache_hits"], counts["cache_misses"], config.similarity_path
        )
    )
    if args.pairs_only or args.no_label_percentiles:
        # A shard run must not label: its percentiles would rank against its own
        # slice rather than the corpus. Run --label-percentiles once at the end.
        if not args.pairs_only:
            log_event("skipped the percentile pass; run --label-percentiles once every shard has finished")
    else:
        labels = label_percentiles(config.similarity_path)
        log_event("percentiles over a population of %d (%d unscored rows left null)" % (labels["population"], labels["unscored"]))

    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
