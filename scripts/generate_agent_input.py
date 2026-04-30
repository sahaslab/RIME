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

def generate_agent_input(plan_prompts_path: str, manifest_path: str, output_path: str):
    manifest = load_records(manifest_path)
    index = load_render_index(manifest)

    plans = load_records(plan_prompts_path)
    outputs = []

    for record in plans:
        output = {}
        key = (record["clip_id"], record["plan_id"])
        artifact_output_path = index.get(key) 
        output["ground_truth_edit_audio"] = artifact_output_path
        output["input_audio"] = record["input_audio"]
        output["prompt_variants"] = record["prompt_variants"]
        output["edit_graph"] = record["edit_graph"]
        outputs.append(output)
    write_jsonl(output_path, outputs)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match plan prompts to render manifest artifacts.")
    parser.add_argument("--plan-prompts",  type=Path, required=True, help="Path to subsampled_plan_prompts.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to render_manifest_*.jsonl")
    parser.add_argument("--output",type=Path,default="matched_prompts.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    generate_agent_input(args.plan_prompts, args.manifest, args.output)