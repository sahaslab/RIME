import re
import ast
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from ground_truth.instrument_vocab import matches_any_word, resolve_instrument_tag, normalize_instrument_tags
import yaml
from typing import Any


class DatasetAdapter(ABC):
    name: str

    @abstractmethod
    def build_manifest(
        self,
        *,
        metadata_dir: Path,
        dataset_config: Mapping[str, Any],
        audio_root: Path | None,
        limit: int | None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError


@dataclass(frozen=True)
class DatasetRegistry:
    adapters: Mapping[str, DatasetAdapter]

    def get(self, name: str) -> DatasetAdapter:
        try:
            return self.adapters[name]
        except KeyError as error:
            raise ValueError(f"Unsupported dataset '{name}'.") from error

    def names(self) -> list[str]:
        return sorted(self.adapters)


class MTGJamendoAdapter(DatasetAdapter):
    name = "mtg_jamendo"

    def build_manifest(
        self,
        *,
        metadata_dir: Path,
        dataset_config: Mapping[str, Any],
        audio_root: Path | None,
        limit: int | None,
        separation_profile: str | None,
    ) -> list[dict[str, Any]]:
        metadata_path = metadata_dir / dataset_config["files"]["metadata"]
        rows: list[dict[str, Any]] = []
        for track_row in _load_mtg_filtered_metadata(metadata_path):
            relative_path = track_row["relative_path"]
            audio_path = _resolve_audio_path(relative_path, audio_root)
            rows.append(
                build_manifest_row(
                    clip_id=track_row["track_id"],
                    audio_path=audio_path,
                    dataset_payload={
                        "name": dataset_config["name"],
                        "track_id": track_row["track_id"],
                        "artist_id": track_row["artist_id"],
                        "album_id": track_row["album_id"],
                        "track_name": track_row["track_id"],
                        "artist_name": track_row["artist_id"],
                        "album_name": track_row["album_id"],
                        "release_date": None,
                        "url": None,
                        "relative_path": relative_path,
                        "duration_s": track_row.get("duration_s"),
                    },
                    genres=track_row["genres"],
                    mood_themes=track_row["mood_themes"],
                    instrument_tags=track_row["instrument_tags"],
                    dataset_config=dataset_config,
                    separation_profile=separation_profile,
                )
            )
            if limit is not None and len(rows) >= limit:
                break
        return rows


class MedleyDBAdapter(DatasetAdapter):
    name = "medleydb"

    def build_manifest(
        self,
        *,
        metadata_dir: Path,
        dataset_config: Mapping[str, Any],
        audio_root: Path | None,
        limit: int | None,
        separation_profile: str | None,
    ) -> list[dict[str, Any]]:
        files = dataset_config["files"]
        rows: list[dict[str, Any]] = []
        metadata_path = metadata_dir / files["metadata"]
        audio_prefix = files.get("audio_subdir", "")
        allowed_types = set(dataset_config.get("allowed_types", []))

        with metadata_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for raw_row in reader:
                if allowed_types and raw_row.get("type") not in allowed_types:
                    continue

                clip_id = str(raw_row.get("medleydb_id") or raw_row["id"])
                relative_audio_path = _join_relative(audio_prefix, raw_row["audio_path"])
                audio_path = _resolve_audio_path(relative_audio_path, audio_root)
                genres = _parse_python_list(raw_row.get("genre"))
                instrument_tags = sorted(set(_parse_quoted_list(raw_row.get("instrument"))))
                key_name, mode_name = _parse_key_mode(raw_row.get("key"))
                tempo_bpm = _parse_tempo_bpm(raw_row.get("tempo"))

                rows.append(
                    build_manifest_row(
                        clip_id=clip_id,
                        audio_path=audio_path,
                        dataset_payload={
                            "name": dataset_config["name"],
                            "track_id": raw_row["id"],
                            "medleydb_id": raw_row.get("medleydb_id"),
                            "track_name": raw_row.get("medleydb_id"),
                            "artist_name": _medleydb_artist_name(raw_row.get("medleydb_id")),
                            "album_name": None,
                            "release_date": None,
                            "url": None,
                            "license": raw_row.get("license"),
                            "type": raw_row.get("type"),
                            "relative_path": relative_audio_path,
                            "duration_s": None,
                        },
                        genres=genres,
                        mood_themes=[],
                        instrument_tags=instrument_tags,
                        dataset_config=dataset_config,
                        separation_profile=separation_profile,
                        tempo_bpm=tempo_bpm,
                        key_name=key_name,
                        mode_name=mode_name,
                    )
                )
                if limit is not None and len(rows) >= limit:
                    break
        return rows


class MusicCapsAdapter(DatasetAdapter):
    name = "musiccaps"

    def build_manifest(
        self,
        *,
        metadata_dir: Path,
        dataset_config: Mapping[str, Any],
        audio_root: Path | None,
        limit: int | None,
        separation_profile: str | None,
    ) -> list[dict[str, Any]]:
        metadata_path = metadata_dir / dataset_config["files"]["metadata"]
        audio_extension = dataset_config.get("audio_extension", ".wav")
        demucs_target_map = dataset_config.get("demucs_target_map", {})
        aspect_domains = dataset_config.get("aspect_domains", {})
        issue_aspects = dataset_config.get("issue_aspects", {})
        stopwords = dataset_config.get("aspect_stopwords", [])
        rows: list[dict[str, Any]] = []

        with metadata_path.open("r", encoding="utf-8") as handle:
            for raw_row in csv.DictReader(handle):
                clip_id = str(raw_row["ytid"])
                relative_path = "%s%s" % (clip_id, audio_extension)
                audio_path = _resolve_audio_path(relative_path, audio_root)

                # MusicCaps ships captions for clips whose YouTube source may no
                # longer be downloadable, so skip rows with no audio on disk.
                if audio_path is not None and not Path(audio_path).is_file():
                    continue

                aspects = _parse_python_list(raw_row.get("aspect_list"))
                instrument_tags, _ = normalize_instrument_tags(aspects, demucs_target_map)
                # An aspect that names an instrument is not also a genre, which
                # keeps "classical guitar" and "electronic drums" out of the
                # genre list, and aspects about the recording itself are skipped
                # so "ambient noises" does not read as the ambient genre.
                genre_aspects = [
                    aspect
                    for aspect in aspects
                    if resolve_instrument_tag(aspect, demucs_target_map) is None and not matches_any_word(aspect, stopwords)
                ]

                rows.append(
                    build_manifest_row(
                        clip_id=clip_id,
                        audio_path=audio_path,
                        dataset_payload={
                            "name": dataset_config["name"],
                            "track_id": clip_id,
                            "track_name": clip_id,
                            "artist_name": None,
                            "album_name": None,
                            "release_date": None,
                            "url": "https://www.youtube.com/watch?v=%s" % clip_id,
                            "relative_path": relative_path,
                            "duration_s": _musiccaps_duration_s(raw_row),
                            "start_s": _parse_float(raw_row.get("start_s")),
                            "end_s": _parse_float(raw_row.get("end_s")),
                            "caption": raw_row.get("caption"),
                            "aspect_list": aspects,
                            "audioset_positive_labels": [
                                label for label in str(raw_row.get("audioset_positive_labels") or "").split(",") if label
                            ],
                            "is_balanced_subset": _parse_bool(raw_row.get("is_balanced_subset")),
                            "is_audioset_eval": _parse_bool(raw_row.get("is_audioset_eval")),
                        },
                        genres=_mine_aspects(genre_aspects, aspect_domains.get("genres", [])),
                        mood_themes=_mine_aspects(aspects, aspect_domains.get("mood_themes", [])),
                        instrument_tags=instrument_tags,
                        dataset_config=dataset_config,
                        separation_profile=separation_profile,
                        issues=_mine_issues(aspects, issue_aspects),
                    )
                )
                if limit is not None and len(rows) >= limit:
                    break
        return rows


DATASET_REGISTRY = DatasetRegistry(
    adapters={
        MTGJamendoAdapter.name: MTGJamendoAdapter(),
        MedleyDBAdapter.name: MedleyDBAdapter(),
        MusicCapsAdapter.name: MusicCapsAdapter(),
    }
)


def available_dataset_names() -> list[str]:
    return DATASET_REGISTRY.names()


def load_dataset_config(path: Path | str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if "dataset" not in loaded:
        raise ValueError("Dataset config '%s' is missing a 'dataset' root key." % config_path)
    dataset = loaded["dataset"] or {}

    # `extends` lets a dataset inherit the shared Demucs stem maps instead of
    # copying them, so the vocabularies cannot drift apart between datasets.
    parent_name = dataset.pop("extends", None)
    if parent_name is None:
        return dataset
    parent_path = (config_path.parent / str(parent_name)).resolve()
    if parent_path == config_path.resolve():
        raise ValueError("Dataset config '%s' extends itself." % config_path)
    return _merge_config(load_dataset_config(parent_path), dataset)


def _merge_config(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge where the overriding config wins on conflicts."""
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_config(existing, value)
        else:
            merged[key] = value
    return merged


def build_analysis_manifest(
    dataset_name: str,
    metadata_dir: Path | str,
    dataset_config_path: Path | str,
    audio_root: Path | str | None = None,
    limit: int | None = None,
    separation_profile: str | None = None,
) -> list[dict[str, Any]]:
    dataset_config = load_dataset_config(dataset_config_path)
    if dataset_name != dataset_config["name"]:
        raise ValueError(
            "Requested dataset '%s' does not match config '%s'."
            % (dataset_name, dataset_config["name"])
        )
    adapter = DATASET_REGISTRY.get(dataset_name)
    return adapter.build_manifest(
        metadata_dir=Path(metadata_dir),
        dataset_config=dataset_config,
        audio_root=None if audio_root is None else Path(audio_root),
        limit=limit,
        separation_profile=separation_profile,
    )


def build_manifest_row(
    *,
    clip_id: str,
    audio_path: str | None,
    dataset_payload: Mapping[str, Any],
    genres: Sequence[str],
    mood_themes: Sequence[str],
    instrument_tags: Sequence[str],
    dataset_config: Mapping[str, Any],
    separation_profile: str | None = None,
    tempo_bpm: float | None = None,
    key_name: str | None = None,
    mode_name: str | None = None,
    issues: Sequence[str] = (),
) -> dict[str, Any]:
    target_candidates = _build_target_candidates(
        instrument_tags=instrument_tags,
        dataset_config=dataset_config,
    )
    return {
        "clip_id": clip_id,
        "audio_path": audio_path,
        "analysis": {
            "dataset": dict(dataset_payload),
            "genres": sorted(set(genres)),
            "mood_themes": sorted(set(mood_themes)),
            "instrument_tags": sorted(set(instrument_tags)),
            "target_candidates": _filter_target_candidates(
                target_candidates,
                separation_profile=separation_profile,
            ),
            "issues": sorted(set(issues)),
            "tempo_bpm": tempo_bpm,
            "key": key_name,
            "mode": mode_name,
        },
    }


def _resolve_audio_path(relative_path: str | None, audio_root: Path | None) -> str | None:
    if relative_path is None or audio_root is None:
        return None
    return str(audio_root / relative_path)


def _join_relative(prefix: str, leaf: str) -> str:
    if not prefix:
        return leaf
    return str(Path(prefix) / leaf)


def _load_mtg_filtered_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t", restkey="EXTRA_TAGS")
        for row in reader:
            tags = [row.get("TAGS", ""), *(row.get("EXTRA_TAGS") or [])]
            rows.append(
                {
                    "track_id": row["TRACK_ID"],
                    "artist_id": row["ARTIST_ID"],
                    "album_id": row["ALBUM_ID"],
                    "relative_path": row["PATH"],
                    "duration_s": float(row["DURATION"]),
                    "instrument_tags": _parse_prefixed_tags(tags, "instrument"),
                    "genres": _parse_prefixed_tags(tags, "genre"),
                    "mood_themes": _parse_prefixed_tags(tags, "mood/theme"),
                }
            )
    return rows


def _build_target_candidates(
    instrument_tags: Sequence[str],
    dataset_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    families = dataset_config.get("target_families", {})
    family_priority = dataset_config.get("family_priority", {})
    stem_map = dataset_config.get("target_stems", {})
    demucs_target_map = dataset_config.get("demucs_target_map", {})
    seen: set[tuple[str, str]] = set()
    candidates: list[dict[str, Any]] = []

    for instrument_tag in instrument_tags:
        separation_target = demucs_target_map.get(instrument_tag)
        if separation_target is None:
            continue
        if separation_target == "other":
            continue
        family = _target_family(separation_target, families)
        stem = stem_map.get(separation_target, separation_target)
        candidate_key = (stem, family)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        candidates.append(
            {
                "stem": stem,
                "family": family,
                "source_tag": instrument_tag,
                "separation_target": separation_target,
                "separation_target_source": "manual_map",
                "priority": float(family_priority.get(family, 1.0)),
            }
        )

    return sorted(
        candidates,
        key=lambda item: (-item["priority"], item["stem"], item["source_tag"]),
    )


def _target_family(instrument_tag: str, families: Mapping[str, Sequence[str]]) -> str:
    lowered = instrument_tag.lower()
    for family_name, family_values in families.items():
        family_tags = {value.lower() for value in family_values}
        if lowered in family_tags:
            return family_name
    return "other"


def _filter_target_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    separation_profile: str | None,
) -> list[dict[str, Any]]:
    if separation_profile in (None, "", "none", "demucs_6s"):
        return [dict(candidate) for candidate in candidates]
    raise ValueError(f"Unsupported separation profile '{separation_profile}'.")


def _parse_prefixed_tags(tags: Sequence[str], domain_name: str) -> list[str]:
    prefix = f"{domain_name}---"
    values: list[str] = []
    for raw_tag in tags:
        normalized = raw_tag.strip()
        if normalized.startswith(prefix):
            values.append(normalized[len(prefix):])
    return sorted(set(values))


def _mine_aspects(aspects: Sequence[str], vocabulary: Sequence[str]) -> list[str]:
    """Collect vocabulary terms that appear as whole words in free-text aspects."""
    found: list[str] = []
    for aspect in aspects:
        found.extend(matches_any_word(aspect, vocabulary))
    return sorted(set(found))


def _mine_issues(aspects: Sequence[str], issue_aspects: Mapping[str, Sequence[str]]) -> list[str]:
    """Map defect phrases in free-text aspects onto the issue names recipes gate on."""
    found: list[str] = []
    for issue_name, phrases in issue_aspects.items():
        for aspect in aspects:
            if matches_any_word(aspect, phrases):
                found.append(issue_name)
                break
    return sorted(set(found))


def _musiccaps_duration_s(row: Mapping[str, Any]) -> float | None:
    start_s = _parse_float(row.get("start_s"))
    end_s = _parse_float(row.get("end_s"))
    if start_s is None or end_s is None:
        return None
    return round(end_s - start_s, 3)


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_python_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if isinstance(parsed, str):
        return [parsed]
    return [str(item) for item in parsed]


def _parse_quoted_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [match.strip() for match in re.findall(r"'([^']+)'", value)]


def _parse_key_mode(value: str | None) -> tuple[str | None, str | None]:
    items = _parse_python_list(value)
    if not items:
        return None, None
    raw = items[0].strip()
    if not raw:
        return None, None
    parts = raw.split()
    if len(parts) >= 2:
        return parts[0], parts[1].lower()
    return raw, None


def _parse_tempo_bpm(value: str | None) -> float | None:
    items = _parse_python_list(value)
    for item in items:
        match = re.search(r"(\d+(?:\.\d+)?)\s*bpm", item, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _medleydb_artist_name(medleydb_id: str | None) -> str | None:
    if not medleydb_id or "_" not in medleydb_id:
        return None
    return medleydb_id.split("_", 1)[0]


def _row_tags(row: Mapping[str, Any]) -> list[str]:
    tags = row.get("TAGS") or ""
    return [value for value in str(tags).split(",") if value]


def _read_tsv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]
