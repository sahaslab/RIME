#!/usr/bin/env python3
"""
tag_instruments.py

Standalone instrument tagger built on Music Flamingo (Audio Flamingo 3).

Given audio files, a directory of audio, or a JSONL manifest with an audio-path
field, the script prompts the model for the instruments present in each clip and
writes one JSONL record per clip:

    {"audio_path": "...", "instruments": ["piano", "drums"], "raw_output": "piano, drums"}

Records are appended and flushed as they are produced, so a long GPU job can be
resumed with --resume after an interruption.

Examples:
    python scripts/tag_instruments.py --audio clip.wav
    python scripts/tag_instruments.py --audio-dir derived/MTG-Jamendo --output derived/instrument_tags.jsonl
    python scripts/tag_instruments.py --input-jsonl derived/ground_truth/subsampled_plans.jsonl --audio-key input_audio --resume
"""

from __future__ import annotations
import ast
import sys
import json
import argparse
from typing import Any
from pathlib import Path
import torch
from transformers import AutoProcessor, AudioFlamingo3ForConditionalGeneration

DEFAULT_MODEL_ID = "nvidia/music-flamingo-hf"
DEFAULT_CACHE_DIR = "/dartfs/rc/lab/S/SinghN/noah/.cache/huggingface/hub"
DEFAULT_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")

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

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class InstrumentTagger:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        cache_dir: str | None = DEFAULT_CACHE_DIR,
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        prompt: str = PROMPT,
        max_new_tokens: int = 1024
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        # Load weights directly in the target dtype so activations stay consistent.
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch_dtype,
            cache_dir=cache_dir
        )
        self.model.eval()

    def process_audio(self, audio_path: str) -> dict[str, Any]:
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
        raw_output = decoded[0].strip()
        return {"instruments": instruments_output_to_list(raw_output), "raw_output": raw_output}


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
            items = [str(item).strip().lower() for item in parsed if str(item).strip()]
            return _dedupe(items)
    except (ValueError, SyntaxError):
        pass

    # Fallback: treat as comma-separated string
    items = [item.strip().lower() for item in text.split(",") if item.strip()]
    return _dedupe(items)


def _dedupe(items: list[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def collect_audio_paths(
    audio: list[Path] | None,
    audio_dir: Path | None,
    input_jsonl: Path | None,
    audio_key: str,
    extensions: tuple[str, ...]
) -> list[str]:
    """Gather audio paths from any combination of the three input modes, deduped in discovery order."""
    paths: list[str] = []

    for path in audio or []:
        paths.append(str(path))

    if audio_dir is not None:
        if not audio_dir.is_dir():
            raise SystemExit("Audio directory not found: %s" % audio_dir)
        suffixes = {extension.lower() for extension in extensions}
        paths.extend(str(path) for path in sorted(audio_dir.rglob("*")) if path.is_file() and path.suffix.lower() in suffixes)

    if input_jsonl is not None:
        if not input_jsonl.is_file():
            raise SystemExit("Input JSONL not found: %s" % input_jsonl)
        with input_jsonl.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    print("Skipping %s line %d: %s" % (input_jsonl, line_number, error), file=sys.stderr)
                    continue
                value = record.get(audio_key)
                if value:
                    paths.append(str(value))
                else:
                    print("Skipping %s line %d: no '%s' key" % (input_jsonl, line_number, audio_key), file=sys.stderr)

    return _dedupe_paths(paths)


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def load_completed_paths(output_path: Path) -> set[str]:
    """Audio paths already tagged successfully in an existing output file."""
    completed: set[str] = set()
    if not output_path.is_file():
        return completed
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("audio_path") and not record.get("error"):
                completed.add(str(record["audio_path"]))
    return completed


def tag_audio_paths(tagger: InstrumentTagger, audio_paths: list[str], output_path: Path, append: bool) -> tuple[int, int]:
    """Tag each clip and append one JSONL record per clip. Returns (tagged, failed)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tagged = 0
    failed = 0
    total = len(audio_paths)

    with output_path.open("a" if append else "w", encoding="utf-8") as handle:
        for index, audio_path in enumerate(audio_paths, start=1):
            record: dict[str, Any] = {"audio_path": audio_path}
            try:
                record.update(tagger.process_audio(audio_path))
                tagged += 1
            except Exception as error:  # keep going; a bad clip should not kill a long job
                record["instruments"] = []
                record["error"] = "%s: %s" % (type(error).__name__, error)
                failed += 1
                print("[%d/%d] FAILED %s -> %s" % (index, total, audio_path, record["error"]), file=sys.stderr)
            else:
                print("[%d/%d] %s -> %s" % (index, total, audio_path, ", ".join(record["instruments"]) or "none"))

            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
            handle.flush()

    return tagged, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag musical instruments in audio clips with Music Flamingo.")
    parser.add_argument("--audio", type=Path, nargs="+", default=None, help="One or more audio files to tag.")
    parser.add_argument("--audio-dir", type=Path, default=None, help="Directory searched recursively for audio files.")
    parser.add_argument("--input-jsonl", type=Path, default=None, help="JSONL manifest containing audio paths.")
    parser.add_argument("--audio-key", default="input_audio", help="Record key holding the audio path in --input-jsonl. Default: %(default)s")
    parser.add_argument("--extensions", nargs="+", default=list(DEFAULT_EXTENSIONS), help="Audio extensions collected by --audio-dir. Default: %(default)s")
    parser.add_argument("--output", type=Path, default=Path("derived/mir/instrument_tags.jsonl"), help="Output JSONL path. Default: %(default)s")
    parser.add_argument("--resume", action="store_true", help="Append to --output and skip clips already tagged without error.")
    parser.add_argument("--limit", type=int, default=None, help="Optional clip limit for smoke tests. Default: no limit")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id. Default: %(default)s")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Hugging Face cache directory. Pass 'none' to use the default cache. Default: %(default)s")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bfloat16", help="Weight dtype. Default: %(default)s")
    parser.add_argument("--device-map", default="auto", help="Accelerate device map passed to from_pretrained. Default: %(default)s")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="Generation cap per clip. Default: %(default)s")
    args = parser.parse_args()

    if args.audio is None and args.audio_dir is None and args.input_jsonl is None:
        parser.error("provide at least one of --audio, --audio-dir, or --input-jsonl")

    audio_paths = collect_audio_paths(
        audio=args.audio,
        audio_dir=args.audio_dir,
        input_jsonl=args.input_jsonl,
        audio_key=args.audio_key,
        extensions=tuple(args.extensions)
    )
    if not audio_paths:
        raise SystemExit("No audio paths found for the given inputs.")

    if args.resume:
        completed = load_completed_paths(args.output)
        if completed:
            audio_paths = [path for path in audio_paths if path not in completed]
            print("Resuming: %d clip(s) already tagged in %s." % (len(completed), args.output))
        if not audio_paths:
            print("Nothing left to tag.")
            return

    if args.limit is not None:
        audio_paths = audio_paths[: args.limit]

    print("Tagging %d clip(s) with %s -> %s" % (len(audio_paths), args.model_id, args.output))
    tagger = InstrumentTagger(
        model_id=args.model_id,
        cache_dir=None if args.cache_dir.lower() == "none" else args.cache_dir,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens
    )

    tagged, failed = tag_audio_paths(tagger, audio_paths, args.output, append=args.resume)
    print("Done: %d tagged, %d failed. Output: %s" % (tagged, failed, args.output))
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
