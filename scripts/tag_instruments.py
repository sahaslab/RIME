#!/usr/bin/env python3
"""
tag_instruments.py

Instrument tagger built on Music Flamingo (Audio Flamingo 3), emitting rows that
drop straight into the ground-truth pipeline.

The model returns free text ("electric guitar, drum kit"), which is normalized
against the `demucs_target_map` vocabulary in the dataset config so the output
matches what `build_analysis_manifest.py` produces from dataset metadata. With
the default `--format manifest`, output is an analysis manifest:

    python scripts/tag_instruments.py --audio-dir derived/MTG-Jamendo --output derived/ground_truth/tagged_analysis_manifest.jsonl
    python scripts/generate_ground_truth_plans.py --analysis-path derived/ground_truth/tagged_analysis_manifest.jsonl

Each manifest row carries `analysis.instrument_tags` (normalized, deduped,
sorted) and the `analysis.target_candidates` derived from them, plus an
`analysis.instrument_tagger` audit block holding the raw model output and any
tags that did not map to the vocabulary.

Use `--format tags` for a lightweight raw-output file instead, and
`--from-tags` to re-derive a manifest from such a file without a GPU (useful
after editing `demucs_target_map`).

Clips that fail to tag are written to a separate `--failures` file rather than
the manifest, so the manifest never contains a clip the model never saw. Re-run
with `--resume` to retry them.

Examples:
    python scripts/tag_instruments.py --audio clip.wav --format tags --output derived/mir/instrument_tags.jsonl
    python scripts/tag_instruments.py --input-jsonl derived/ground_truth/mtg_jamendo_analysis_manifest.jsonl --output derived/ground_truth/tagged_analysis_manifest.jsonl --resume
    python scripts/tag_instruments.py --from-tags derived/mir/instrument_tags.jsonl --output derived/ground_truth/tagged_analysis_manifest.jsonl
"""

from __future__ import annotations
import ast
import sys
import json
import argparse
from typing import Any
from pathlib import Path
from dataclasses import field, dataclass
from collections.abc import Mapping, Sequence
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_truth.datasets import load_dataset_config, build_manifest_row
from ground_truth.instrument_vocab import dedupe, normalize_instrument_tags

DEFAULT_MODEL_ID = "nvidia/music-flamingo-hf"
DEFAULT_CACHE_DIR = "/dartfs/rc/lab/S/SinghN/noah/.cache/huggingface/hub"
DEFAULT_DATASET_CONFIG = Path("configs/ground_truth/datasets/mtg_jamendo.yaml")
DEFAULT_OUTPUT_PATH = Path("derived/ground_truth/tagged_analysis_manifest.jsonl")
DEFAULT_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")
AUDIO_PATH_KEYS = ("audio_path", "input_audio", "path", "audio")

PROMPT = """You are an instrument-tagging system.
Task: identify all musical instruments present in the audio.
Output rules (must follow):

Output exactly one line.

Output only a comma-and-space separated list of instrument names (example: piano, bass guitar, drums).

No extra words, labels, punctuation (besides commas), explanations, or newlines.

Do not use "and".

Use lowercase, singular instrument names.

Remove duplicates.
If no instruments are detected, output none."""


@dataclass
class Clip:
    """One unit of work: an audio file plus whatever context the input carried."""

    audio_path: str
    clip_id: str
    source_record: dict[str, Any] = field(default_factory=dict)


class InstrumentTagger:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        prompt: str = PROMPT,
        max_new_tokens: int = 1024
    ) -> None:
        # Imported here so --from-tags and --help do not pay for the GPU stack.
        import torch
        from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration

        self.torch = torch
        self.model_id = model_id
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        # Load weights directly in the target dtype so activations stay consistent.
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=getattr(torch, dtype),
            cache_dir=cache_dir
        )
        self.model.eval()

    def process_audio(self, audio_path: str) -> str:
        """Return the model's raw one-line instrument listing for one clip."""
        torch = self.torch
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "audio", "path": audio_path}
                ]
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True
        )

        # Move to device, and cast float tensors to the model dtype,
        # while leaving integer tensors (input_ids, attention_mask) alone.
        device = self.model.device
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                value = value.to(device)
                if value.is_floating_point():
                    value = value.to(dtype=self.model.dtype)
                inputs[key] = value

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        decoded = self.processor.batch_decode(outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return decoded[0].strip()


def instruments_output_to_list(model_output: str) -> list[str]:
    """
    Convert model output into a Python list of instruments.

    Handles formats like:
    - "piano, bass guitar, drums"
    - ["bass", "guitar"]
    - '["bass", "guitar"]'
    - none
    """

    if not model_output:
        return []

    text = model_output.strip()

    # Handle "none"
    if text.lower() == "none":
        return []

    # Try parsing as a Python list (handles ["bass", "guitar"])
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return dedupe(str(item).strip().lower() for item in parsed if str(item).strip())
    except (ValueError, SyntaxError):
        pass

    # Fallback: treat as comma-separated string
    return dedupe(item.strip().lower() for item in text.split(",") if item.strip())


def build_tagged_manifest_row(
    clip: Clip,
    raw_output: str,
    dataset_config: Mapping[str, Any],
    separation_profile: str | None,
    audio_root: Path | None,
    model_id: str,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """Build an analysis-manifest row whose instrument tags come from the model."""
    raw_tags = instruments_output_to_list(raw_output)
    demucs_target_map = dataset_config.get("demucs_target_map", {})
    instrument_tags, unmatched = normalize_instrument_tags(raw_tags, demucs_target_map)

    analysis = dict(clip.source_record.get("analysis") or {})
    dataset_payload = dict(analysis.get("dataset") or {})
    dataset_payload.setdefault("name", dataset_name or dataset_config.get("name"))
    dataset_payload.setdefault("track_id", clip.clip_id)
    dataset_payload.setdefault("track_name", clip.clip_id)
    dataset_payload.setdefault("relative_path", _relative_path(clip.audio_path, audio_root))
    for optional_key in ("artist_name", "album_name", "release_date", "url", "duration_s"):
        dataset_payload.setdefault(optional_key, None)

    row = build_manifest_row(
        clip_id=clip.clip_id,
        audio_path=clip.audio_path,
        dataset_payload=dataset_payload,
        genres=analysis.get("genres") or [],
        mood_themes=analysis.get("mood_themes") or [],
        instrument_tags=instrument_tags,
        dataset_config=dataset_config,
        separation_profile=separation_profile,
        tempo_bpm=analysis.get("tempo_bpm"),
        key_name=analysis.get("key"),
        mode_name=analysis.get("mode"),
        issues=analysis.get("issues") or ()
    )
    row["analysis"]["instrument_tagger"] = {
        "model_id": model_id,
        "raw_output": raw_output,
        "raw_tags": raw_tags,
        "unmatched_tags": unmatched
    }
    return row


def build_tag_row(clip: Clip, raw_output: str, dataset_config: Mapping[str, Any], model_id: str) -> dict[str, Any]:
    """Build a lightweight raw-output row for inspection or later --from-tags runs."""
    raw_tags = instruments_output_to_list(raw_output)
    instrument_tags, unmatched = normalize_instrument_tags(raw_tags, dataset_config.get("demucs_target_map", {}))
    return {
        "clip_id": clip.clip_id,
        "audio_path": clip.audio_path,
        "instrument_tags": instrument_tags,
        "raw_tags": raw_tags,
        "unmatched_tags": unmatched,
        "raw_output": raw_output,
        "model_id": model_id
    }


def _relative_path(audio_path: str, audio_root: Path | None) -> str:
    if audio_root is not None:
        try:
            return str(Path(audio_path).relative_to(audio_root))
        except ValueError:
            pass
    return Path(audio_path).name


def collect_clips(
    audio: Sequence[Path] | None,
    audio_dir: Path | None,
    input_jsonl: Path | None,
    audio_key: str | None,
    clip_id_key: str,
    extensions: Sequence[str]
) -> list[Clip]:
    """Gather work items from any combination of the input modes, deduped by audio path."""
    clips: list[Clip] = []

    for path in audio or []:
        clips.append(Clip(audio_path=str(path), clip_id=Path(path).stem))

    if audio_dir is not None:
        if not audio_dir.is_dir():
            raise SystemExit("Audio directory not found: %s" % audio_dir)
        suffixes = {extension.lower() for extension in extensions}
        for path in sorted(audio_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in suffixes:
                clips.append(Clip(audio_path=str(path), clip_id=path.stem))

    if input_jsonl is not None:
        clips.extend(_clips_from_jsonl(input_jsonl, audio_key=audio_key, clip_id_key=clip_id_key))

    return _dedupe_clips(clips)


def _clips_from_jsonl(path: Path, audio_key: str | None, clip_id_key: str) -> list[Clip]:
    clips: list[Clip] = []
    for line_number, record in _iter_jsonl(path):
        resolved_key = audio_key or _detect_audio_key(record)
        value = record.get(resolved_key) if resolved_key else None
        if not value:
            print("Skipping %s line %d: no audio path (looked for %s)" % (path, line_number, audio_key or "/".join(AUDIO_PATH_KEYS)), file=sys.stderr)
            continue
        clip_id = record.get(clip_id_key) or Path(str(value)).stem
        clips.append(Clip(audio_path=str(value), clip_id=str(clip_id), source_record=record))
    return clips


def _detect_audio_key(record: Mapping[str, Any]) -> str | None:
    for key in AUDIO_PATH_KEYS:
        if record.get(key):
            return key
    return None


def _iter_jsonl(path: Path):
    if not path.is_file():
        raise SystemExit("Input JSONL not found: %s" % path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                print("Skipping %s line %d: %s" % (path, line_number, error), file=sys.stderr)


def _dedupe_clips(clips: Sequence[Clip]) -> list[Clip]:
    seen: set[str] = set()
    result: list[Clip] = []
    for clip in clips:
        if clip.audio_path not in seen:
            seen.add(clip.audio_path)
            result.append(clip)
    return result


def load_completed_paths(output_path: Path) -> set[str]:
    """Audio paths already tagged without error in an existing output file."""
    completed: set[str] = set()
    if not output_path.is_file():
        return completed
    for _, record in _iter_jsonl(output_path):
        if record.get("audio_path") and not record.get("error"):
            completed.add(str(record["audio_path"]))
    return completed


def write_rows(output_path: Path, rows, append: bool) -> None:
    """Append rows one at a time, flushing so a long job stays resumable."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a" if append else "w", encoding="utf-8") as handle:
        for row in rows:
            _write_row(handle, row)


def _write_row(handle, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, sort_keys=True))
    handle.write("\n")
    handle.flush()


def run_tagging(
    tagger: InstrumentTagger,
    clips: Sequence[Clip],
    row_builder,
    output_path: Path,
    failures_path: Path,
    append: bool
) -> dict[str, int]:
    """
    Tag each clip, writing successes to output_path and failures to failures_path.

    Failures are kept out of the output on purpose: a row with no instrument tags
    is still a valid manifest row, so the planner would happily emit whole-mix
    plans for a clip that was never actually tagged.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"tagged": 0, "failed": 0}
    total = len(clips)
    mode = "a" if append else "w"

    with output_path.open(mode, encoding="utf-8") as output_handle, failures_path.open(mode, encoding="utf-8") as failures_handle:
        for index, clip in enumerate(clips, start=1):
            try:
                raw_output = tagger.process_audio(clip.audio_path)
            except Exception as error:  # keep going; a bad clip should not kill a long job
                counts["failed"] += 1
                message = "%s: %s" % (type(error).__name__, error)
                print("[%d/%d] FAILED %s -> %s" % (index, total, clip.audio_path, message), file=sys.stderr)
                _write_row(failures_handle, {"clip_id": clip.clip_id, "audio_path": clip.audio_path, "error": message})
                continue

            counts["tagged"] += 1
            row = row_builder(clip, raw_output)
            tags = row.get("instrument_tags") or row.get("analysis", {}).get("instrument_tags", [])
            print("[%d/%d] %s -> %s" % (index, total, clip.audio_path, ", ".join(tags) or "none"))
            _write_row(output_handle, row)

    return counts


def iter_rows_from_tags(records, row_builder, clip_id_key: str):
    """Re-derive rows from a previously written tags file, no model needed."""
    for _, record in records:
        audio_path = record.get("audio_path")
        if not audio_path:
            continue
        clip = Clip(
            audio_path=str(audio_path),
            clip_id=str(record.get(clip_id_key) or Path(str(audio_path)).stem),
            source_record=record
        )
        yield row_builder(clip, str(record.get("raw_output") or ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tag musical instruments in audio clips with Music Flamingo, in ground-truth manifest format.")
    parser.add_argument("--audio", type=Path, nargs="+", default=None, help="One or more audio files to tag.")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Directory searched recursively for audio files.")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="JSONL of analysis-manifest or plan rows to tag, whose clip_id and analysis fields are carried through.")
    parser.add_argument("--from-tags", type=Path, default=None, help="Re-derive output from a --format tags file without loading the model.")
    parser.add_argument("--audio-key", default=None, help="Record key holding the audio path in JSONL input. Default: first present of %s" % "/".join(AUDIO_PATH_KEYS))
    parser.add_argument("--clip-id-key", default="clip_id", help="Record key holding the clip id in JSONL input. Default: %(default)s")
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS), help="Audio extensions collected by --audio-dir. Default: %(default)s")
    parser.add_argument("--format", choices=["manifest", "tags"], default="manifest", help="manifest: an analysis manifest. tags: raw per-clip model output. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output JSONL path. Default: %(default)s")
    parser.add_argument("--failures", type=Path, default=None, help="JSONL path for clips that failed to tag, kept out of the manifest. Default: <output>.failures.jsonl")
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG, help="Dataset config supplying the instrument vocabulary and stem maps. Default: %(default)s")
    parser.add_argument("--dataset-name", default=None, help="Value for analysis.dataset.name, for tagging a corpus with another dataset's vocabulary. Default: the config's own name")
    parser.add_argument("--audio-root", type=Path, default=None, help="Root used to compute analysis.dataset.relative_path. Default: the file name alone")
    parser.add_argument("--separation-profile", choices=["demucs_6s", "none"], default="demucs_6s", help="Target filtering profile, matching build_analysis_manifest.py. Default: %(default)s")
    parser.add_argument("--resume", action="store_true", help="Append to --output and skip clips it already contains, so previously failed clips are retried.")
    parser.add_argument("--limit", type=int, default=None, help="Optional clip limit for smoke tests. Default: no limit")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id. Default: %(default)s")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face cache directory. Pass 'none' to use the default cache. Default: %(default)s")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16", help="Weight dtype. Default: %(default)s")
    parser.add_argument("--device-map", default="auto", help="Accelerate device map passed to from_pretrained. Default: %(default)s")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation cap per clip. Default: %(default)s")
    args = parser.parse_args()

    if args.from_tags is None and args.audio is None and args.audio_dir is None and args.input_jsonl is None:
        parser.error("provide at least one of --audio, --audio-dir, --input-jsonl, or --from-tags")
    if args.from_tags is not None and (args.audio or args.audio_dir or args.input_jsonl):
        parser.error("--from-tags cannot be combined with audio inputs")
    if args.failures is None:
        args.failures = args.output.with_suffix(args.output.suffix + ".failures.jsonl")
    return args


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_config(args.dataset_config)
    separation_profile = None if args.separation_profile == "none" else args.separation_profile

    if args.format == "manifest":
        def row_builder(clip: Clip, raw_output: str) -> dict[str, Any]:
            return build_tagged_manifest_row(
                clip=clip,
                raw_output=raw_output,
                dataset_config=dataset_config,
                separation_profile=separation_profile,
                audio_root=args.audio_root,
                model_id=args.model_id,
                dataset_name=args.dataset_name
            )
    else:
        def row_builder(clip: Clip, raw_output: str) -> dict[str, Any]:
            return build_tag_row(clip, raw_output, dataset_config, args.model_id)

    if args.from_tags is not None:
        rows = iter_rows_from_tags(_iter_jsonl(args.from_tags), row_builder, args.clip_id_key)
        write_rows(args.output, rows, append=False)
        print("Rebuilt %s from %s" % (args.output, args.from_tags))
        return

    clips = collect_clips(
        audio=args.audio,
        audio_dir=args.audio_dir,
        input_jsonl=args.input_jsonl,
        audio_key=args.audio_key,
        clip_id_key=args.clip_id_key,
        extensions=args.extensions
    )
    if not clips:
        raise SystemExit("No audio paths found for the given inputs.")

    if args.resume:
        completed = load_completed_paths(args.output)
        if completed:
            clips = [clip for clip in clips if clip.audio_path not in completed]
            print("Resuming: %d clip(s) already tagged in %s." % (len(completed), args.output))
        if not clips:
            print("Nothing left to tag.")
            return

    if args.limit is not None:
        clips = clips[: args.limit]

    print("Tagging %d clip(s) with %s -> %s (%s format)" % (len(clips), args.model_id, args.output, args.format))
    tagger = InstrumentTagger(
        model_id=args.model_id,
        cache_dir=None if args.cache_dir.lower() == "none" else args.cache_dir,
        dtype=args.dtype,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens
    )

    counts = run_tagging(
        tagger=tagger,
        clips=clips,
        row_builder=row_builder,
        output_path=args.output,
        failures_path=args.failures,
        append=args.resume
    )
    print("Done: %d tagged, %d failed. Output: %s" % (counts["tagged"], counts["failed"], args.output))
    if counts["failed"]:
        print("Failures recorded in %s; re-run with --resume to retry them." % args.failures, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
