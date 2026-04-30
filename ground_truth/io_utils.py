import json
from pathlib import Path
from collections.abc import Iterable
import yaml
from typing import Any


# Map supported file extensions to load handlers.
def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    handlers = {
        ".jsonl": _load_jsonl_records,
        ".json": _load_json_records,
        ".yaml": _load_yaml_records,
        ".yml": _load_yaml_records
    }
    if suffix not in handlers:
        raise ValueError("Unsupported file type '%s'." % suffix)
    return handlers[suffix](path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return _normalize_loaded_records(json.load(handle))


def _load_yaml_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return _normalize_loaded_records(yaml.safe_load(handle))


def _normalize_loaded_records(loaded: Any) -> list[dict[str, Any]]:
    if loaded is None:
        return []
    return loaded if isinstance(loaded, list) else [loaded]
