#!/usr/bin/env python3
"""
Check which output_path entries in a JSONL file don't exist on disk.
Usage: python check_missing_files.py <path_to_jsonl>
"""

import json
import sys
from pathlib import Path


def check_missing_files(jsonl_path: str) -> None:
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"Error: JSONL file not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)

    missing = []
    errors = []

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"  Line {line_num}: JSON parse error — {e}")
                continue

            output_path = record.get("output_path")
            if output_path is None:
                errors.append(f"  Line {line_num}: no 'output_path' key")
                continue

            if not Path(output_path).exists():
                missing.append((line_num, output_path))

    # Report
    if missing:
        print(f"Missing files ({len(missing)}):")
        for line_num, path in missing:
            print(f"  Line {line_num}: {path}")
    else:
        print("All output_path files exist.")

    if errors:
        print(f"\nSkipped lines ({len(errors)}):")
        for msg in errors:
            print(msg)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_jsonl>", file=sys.stderr)
        sys.exit(1)

    check_missing_files(sys.argv[1])