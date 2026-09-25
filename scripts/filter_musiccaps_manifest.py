import os
import json
import random 
from pathlib import Path

SHARED_ROOT = Path(os.environ.get("RIME_SHARED_ROOT", "data/shared")).expanduser()
ARTIFACTS_ROOT = Path(os.environ.get("RIME_ARTIFACTS_ROOT", "derived")).expanduser()
BASE_PATH = SHARED_ROOT / "musiccaps/rime-metadata/musiccaps_analysis_manifest.jsonl"
WRITE_PATH = ARTIFACTS_ROOT / "manifests"
 
def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def filter_and_sample(path: str, held_out: int) -> list[dict]:
    targets = ["bass", "drums", "vocals", "guitar"]
    filtered = []

    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e

            instruments = [a["separation_target"] for a in record["analysis"]["target_candidates"]]
            if len(instruments) > 1 and any(stem in targets for stem in instruments):
                filtered.append(record)

    if held_out > len(filtered):
        raise ValueError(f"held_out={held_out} but only {len(filtered)} records matched")

    valid_idx = set(random.sample(range(len(filtered)), k=held_out))
    valid = [r for i, r in enumerate(filtered) if i in valid_idx]
    train = [r for i, r in enumerate(filtered) if i not in valid_idx]

    

    return train, valid


if __name__ == "__main__":
    random.seed(18261)
    train,valid = filter_and_sample(BASE_PATH,399)
    out_dir = Path(WRITE_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "musiccaps_analysis_manifest.jsonl", train)
    write_jsonl(out_dir / "musiccaps_heldout_analysis_manifest.jsonl", valid)

    print(f"train: {len(train)}  valid: {len(valid)}  -> {out_dir}")