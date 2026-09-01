"""
match_artifacts.py

Matches entries in subsampled_plan_prompts.jsonl to their corresponding
artifact in render_manifest_*.jsonl using:
  - plan_prompts["clip_id"]        <-> render_manifest["clip_id"]
  - plan_prompts["plan_id"] <-> render_manifest["plan_id"]

Outputs a JSONL file where each line is the plan-prompt record enriched
with a new "output_path" field containing the rendered .wav path.
Unmatched plan-prompt rows are written with "artifact_dir": null.
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

def generate_agent_input(plan_prompts_path: str, manifest_path: str, output_path: str, poisoning: bool=False, training: bool=False):
    if not training: 
        manifest = load_records(manifest_path)
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
        
        if poisoning:
            gt_path, baseline_path = index.get(key, (None, None))
            output["ground_truth_edit_audio"] = gt_path
            record["original_clean_audio"] = record["input_audio"]
            output["input_audio"] = baseline_path
        if training:
            output["ground_truth_edit_audio"] = ""
            output["input_audio"] = record["input_audio"]
        else:
            artifact_output_path = index.get(key) 
            output["ground_truth_edit_audio"] = artifact_output_path
            output["input_audio"] = record["input_audio"]
        output["prompt_variants"] = record["prompt_variants"]
        output["metadata"] = record
        outputs.append(output)
    write_jsonl(output_path, outputs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match plan prompts to render manifest artifacts.")
    parser.add_argument("--plan-prompts",  type=Path, required=True, help="Path to subsampled_plan_prompts.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to render_manifest_*.jsonl")
    parser.add_argument("--output",type=Path,default="matched_prompts.jsonl", help="Output JSONL path")
    parser.add_argument("--poisoning", action="store_true", help="Use poisoning logic")
    parser.add_argument("--training", action="store_true", help="Use poisoning logic")
    args = parser.parse_args()

    generate_agent_input(args.plan_prompts, args.manifest, args.output, args.poisoning, args.training)