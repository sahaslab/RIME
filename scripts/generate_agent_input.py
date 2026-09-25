"""
generate_agent_input.py

Matches entries in subsampled_plan_prompts.jsonl to their corresponding
artifact in render_manifest_*.jsonl using:
  - plan_prompts["clip_id"]        <-> render_manifest["clip_id"]
  - plan_prompts["plan_id"] <-> render_manifest["plan_id"]

Outputs a JSONL file where each line is the plan-prompt record enriched
with a new "output_path" field containing the rendered .wav path.
Plan-prompt rows with no successful render (missing or "status": "error") are dropped.

Three mutually exclusive modes:

  default      pair each prompt with its rendered edit from the manifest.
  --poisoning  the manifest also carries a poisoned baseline. The agent is fed
               that baseline as `input_audio`, and the clean original stays on
               the record as `original_clean_audio`.
  --training   there is no rendered audio, so there is no manifest to read:
               `ground_truth_edit_audio` comes out empty and `input_audio` is
               carried through untouched. Nothing here opens an audio file, so
               training rows can be built without the corpus mounted.
"""

import json
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_truth.io_utils import load_records, write_jsonl


def load_render_index(manifest: list[dict]) -> dict:
    """Build a lookup dict keyed by (clip_id, plan_id) -> output_path."""
    return {
        (record["clip_id"], record["plan_id"]): record.get("output_path")
        for record in manifest
    }

def load_render_poison_index(manifest: list[dict]) -> dict:
    return {
        (record["clip_id"], record["plan_id"]): (record.get("output_path"), record.get("baseline_path"))
        for record in manifest
    }

def generate_agent_input(plan_prompts_path: str, manifest_path: str | None, output_path: str, poisoning: bool=False, training: bool=False):
    # The two flags answer the same question -- what goes in `input_audio` --
    # with different answers, so taking both is a mistake rather than a
    # combination to resolve silently.
    if training and poisoning:
        raise ValueError(
            "--training and --poisoning are mutually exclusive: training pairs a prompt with no rendered "
            "audio at all, poisoning pairs it with a poisoned render."
        )
    if not training:
        if manifest_path is None:
            raise ValueError(
                "--manifest is required unless --training is passed: it is what supplies the rendered audio "
                "paths each prompt is paired with."
            )
        manifest = [r for r in load_records(manifest_path) if r.get("status") != "error"]
        if poisoning:
            index = load_render_poison_index(manifest)
        else: 
            index = load_render_index(manifest)

    plans = load_records(plan_prompts_path)
    outputs = []

    for record in plans:
        output = {}

        if not training:
            key = (record["clip_id"], record["plan_id"])
            if key not in index:
                continue
        
        # One branch per mode, and no fallthrough: as an `if poisoning` followed
        # by an `if training / else`, a poisoning run fell into the else and had
        # both of its assignments overwritten by the default ones.
        if training:
            output["ground_truth_edit_audio"] = ""
            output["input_audio"] = record["input_audio"]
        elif poisoning:
            # `key` is in the index -- a miss was skipped above -- so indexing
            # rather than .get() keeps a future miss loud instead of writing a
            # row with a null path.
            gt_path, baseline_path = index[key]
            output["ground_truth_edit_audio"] = gt_path
            record["original_clean_audio"] = record["input_audio"]
            output["input_audio"] = baseline_path
        else:
            output["ground_truth_edit_audio"] = index[key]
            output["input_audio"] = record["input_audio"]
        output["prompt_variants"] = record["prompt_variants"]
        output["metadata"] = record
        outputs.append(output)
    write_jsonl(output_path, outputs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match plan prompts to render manifest artifacts.")
    parser.add_argument("--plan-prompts",  type=Path, required=True, help="Path to subsampled_plan_prompts.jsonl")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to render_manifest_*.jsonl. Required unless --training, which reads no manifest.")
    parser.add_argument("--output",type=Path,default="matched_prompts.jsonl", help="Output JSONL path")
    parser.add_argument("--poisoning", action="store_true", help="Feed the poisoned baseline as input_audio and keep the clean original as original_clean_audio.")
    parser.add_argument("--training", action="store_true", help="Emit rows with no rendered audio: no manifest is read and ground_truth_edit_audio is empty.")
    args = parser.parse_args()

    generate_agent_input(args.plan_prompts, args.manifest, args.output, args.poisoning, args.training)