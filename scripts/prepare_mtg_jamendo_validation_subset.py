import sys
import argparse
import ast
import csv
import hashlib
import json
import random
import requests
import tarfile
import tempfile
import soundfile
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_truth.datasets import build_analysis_manifest, load_dataset_config
from ground_truth.io_utils import write_json, write_jsonl
from ground_truth.planner import GroundTruthPlanner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("derived/validation/mtg_jamendo_10"), help="Validation workspace root. Default: %(default)s")
    parser.add_argument("--dataset-root", type=Path, default=Path("mtg-jamendo-dataset"), help="Local MTG-Jamendo dataset repository root. Default: %(default)s")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Local metadata TSV. Default: <dataset-root>/data/raw_30s_cleantags.tsv")
    parser.add_argument("--download-from", choices=["mtg", "mtg-fast"], default="mtg-fast", help="Full-quality MTG tar mirror. Default: %(default)s")
    parser.add_argument("--tar-mode", choices=["stream", "cache"], default="stream", help="How to read full-quality tars. stream stops after the target member; cache downloads complete tars. Default: %(default)s")
    parser.add_argument("--tar-cache-dir", type=Path, default=None, help="Directory for downloaded full-quality tars. Default: <output-root>/tar_cache")
    parser.add_argument("--num-tracks", type=int, default=10, help="Number of validation tracks. Default: %(default)s")
    parser.add_argument("--candidate-scan-limit", type=int, default=1500, help="Number of streaming rows to scan before stratified selection. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=0, help="Selection seed. Default: %(default)s")
    parser.add_argument("--max-per-artist", type=int, default=1, help="Preferred maximum selected tracks per artist before relaxing. Default: %(default)s")
    parser.add_argument("--max-per-album", type=int, default=1, help="Preferred maximum selected tracks per album before relaxing. Default: %(default)s")
    parser.add_argument("--required-target-family", action="append", default=["vocals", "drums"], help="Required target family for selected tracks. Repeatable. Default: vocals and drums")
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/ground_truth/datasets/mtg_jamendo.yaml"), help="MTG-Jamendo ground-truth dataset config. Default: %(default)s")
    parser.add_argument("--config-dir", type=Path, default=Path("configs/ground_truth"), help="Ground-truth planner config directory. Default: %(default)s")
    parser.add_argument("--max-variants-per-recipe", type=int, default=None, help="Optional hard cap per recipe per clip. Default: no cap")
    parser.add_argument("--variant-selection", choices=["diverse", "random"], default="random", help="Variant selection when capped. Default: %(default)s")
    parser.add_argument("--random-plans-per-clip", type=int, default=8, help="Additional constrained random plans per clip. Default: %(default)s")
    parser.add_argument("--skip-audio", action="store_true", help="Write metadata and plans without materializing audio files. Default: download/write audio")
    parser.add_argument("--force-audio", action="store_true", help="Overwrite existing selected audio files. Default: reuse existing files")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    local_metadata_path = (
        args.metadata_path.expanduser().resolve()
        if args.metadata_path is not None
        else dataset_root / "data" / "raw_30s_cleantags.tsv"
    )
    tar_cache_dir = (
        args.tar_cache_dir.expanduser().resolve()
        if args.tar_cache_dir is not None
        else output_root / "tar_cache"
    )

    dataset_config = load_dataset_config(args.dataset_config)
    candidates = load_candidates(
        dataset_root=dataset_root,
        metadata_path=local_metadata_path,
        dataset_config=dataset_config,
        scan_limit=args.candidate_scan_limit,
        min_candidates=args.num_tracks,
        seed=args.seed,
        required_target_families=args.required_target_family,
    )
    selected = select_stratified(
        candidates=candidates,
        limit=args.num_tracks,
        seed=args.seed,
        max_per_artist=args.max_per_artist,
        max_per_album=args.max_per_album,
    )

    if not args.skip_audio:
        write_audio_files(
            selected=selected,
            output_root=output_root,
            force_audio=args.force_audio,
            dataset_root=dataset_root,
            tar_cache_dir=tar_cache_dir,
            download_from=args.download_from,
            tar_mode=args.tar_mode,
        )

    metadata_path = write_filtered_metadata(
        selected=selected,
        output_root=output_root,
    )
    selected_tracks_path = output_root / "selected_tracks.jsonl"
    write_jsonl(
        selected_tracks_path,
        selected_track_rows(selected),
    )

    analysis_rows = build_analysis_manifest(
        dataset_name="mtg_jamendo",
        metadata_dir=metadata_path.parent,
        dataset_config_path=args.dataset_config,
        audio_root=None if args.skip_audio else output_root,
        limit=None,
        separation_profile="demucs_6s",
    )
    analysis_path = output_root / "analysis_manifest.jsonl"
    write_jsonl(analysis_path, analysis_rows)

    plans_path = output_root / "all_plans.jsonl"
    plan_count = write_all_plans(
        analysis_rows=analysis_rows,
        plans_path=plans_path,
        config_dir=args.config_dir,
        max_variants_per_recipe=args.max_variants_per_recipe,
        variant_selection=args.variant_selection,
        random_plans_per_clip=args.random_plans_per_clip,
        seed=args.seed,
    )

    summary_path = output_root / "summary.json"
    write_json(
        summary_path,
        build_summary(
            selected=selected,
            output_root=output_root,
            selected_tracks_path=selected_tracks_path,
            metadata_path=metadata_path,
            analysis_path=analysis_path,
            plans_path=plans_path,
            plan_count=plan_count,
        ),
    )

    print("Wrote MTG-Jamendo validation subset to %s" % output_root)
    print("Tracks: %d | plans: %d" % (len(selected), plan_count))
    print("Manifest: %s" % analysis_path)
    print("Plans: %s" % plans_path)
    print("Render with: bash scripts/run_mtg_jamendo_validation_loop.sh")


def load_candidates(
    dataset_root: Path,
    metadata_path: Path,
    dataset_config: Mapping[str, Any],
    scan_limit: int,
    min_candidates: int,
    seed: int,
    required_target_families: Sequence[str],
) -> list[dict[str, Any]]:
    required_families = {str(value) for value in required_target_families}
    sha256_tracks = load_track_sha256(dataset_root / "data" / "download" / "raw_30s_audio_sha256_tracks.txt")
    candidates: list[dict[str, Any]] = []
    rows_scanned = 0
    print(
        "Using local MTG-Jamendo metadata scanner | metadata=%s | minimum_scan_rows=%d | requested_candidates=%d | required_families=%s"
        % (metadata_path, scan_limit, min_candidates, sorted(required_families))
    )
    progress = tqdm(total=None, desc="Reading local MTG-Jamendo metadata", unit="row")
    for row_index, row in enumerate(iter_local_metadata_rows(metadata_path)):
        rows_scanned = row_index + 1
        if row_index >= scan_limit and len(candidates) >= min_candidates:
            break
        candidate = normalize_local_metadata_row(
            row=row,
            dataset_config=dataset_config,
            seed=seed,
            sha256_tracks=sha256_tracks,
        )
        if candidate["target_candidates"] and required_families.issubset(set(candidate["target_families"])):
            candidates.append(candidate)
        progress.update(1)
    progress.close()
    print(
        "Local metadata scan complete | rows_scanned=%d | compatible_candidates=%d"
        % (rows_scanned, len(candidates))
    )
    if len(candidates) == 0:
        raise ValueError("No compatible MTG-Jamendo candidates found with required families %s." % sorted(required_families))
    if len(candidates) < min_candidates:
        raise ValueError(
            "Only found %d compatible MTG-Jamendo candidates after scanning %d rows; requested %d."
            % (len(candidates), rows_scanned, min_candidates)
        )
    return candidates


def iter_local_metadata_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t", restkey="EXTRA_TAGS")
        return [dict(row) for row in reader]


def normalize_local_metadata_row(
    row: Mapping[str, Any],
    dataset_config: Mapping[str, Any],
    seed: int,
    sha256_tracks: Mapping[str, str],
) -> dict[str, Any]:
    track_id = str(row["TRACK_ID"])
    artist_id = str(row["ARTIST_ID"])
    album_id = str(row["ALBUM_ID"])
    numeric_track_id = clean_int(track_id)
    tags = [row.get("TAGS", ""), *(row.get("EXTRA_TAGS") or [])]
    instruments = parse_prefixed_tags(tags, "instrument")
    genres = parse_prefixed_tags(tags, "genre")
    mood_themes = parse_prefixed_tags(tags, "mood/theme")
    target_candidates = build_target_candidates(
        instrument_tags=instruments,
        dataset_config=dataset_config,
    )
    source_path = str(row["PATH"])
    return {
        "track_id": track_id,
        "numeric_track_id": numeric_track_id,
        "artist_id": artist_id,
        "album_id": album_id,
        "relative_path": "audio/%s.wav" % track_id,
        "duration_s": float(row.get("DURATION", 30.0)),
        "genres": genres,
        "mood_themes": mood_themes,
        "instrument_tags": instruments,
        "target_candidates": target_candidates,
        "target_families": sorted({candidate["family"] for candidate in target_candidates}),
        "target_stems": sorted({candidate["stem"] for candidate in target_candidates}),
        "source_path": source_path,
        "source_sha256": sha256_tracks.get(source_path),
        "_tar_name": tar_name_for_source_path(source_path),
        "_noise": stable_noise(seed, numeric_track_id),
    }


def load_track_sha256(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            pieces = line.strip().split()
            if len(pieces) >= 2:
                checksums[pieces[1]] = pieces[0]
    return checksums


def tar_name_for_source_path(source_path: str) -> str:
    prefix = Path(source_path).parts[0]
    return "raw_30s_audio-%s.tar" % prefix


def parse_prefixed_tags(tags: Sequence[str], domain_name: str) -> list[str]:
    prefix = "%s---" % domain_name
    values = []
    for raw_tag in tags:
        normalized = str(raw_tag).strip()
        if normalized.startswith(prefix):
            values.append(normalized[len(prefix):])
    return sorted(set(values))


def clean_int(value: Any) -> int:
    digits = "".join(ch for ch in str(value).replace(",", "").strip() if ch.isdigit())
    return int(digits)


def normalize_tag_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    if isinstance(value, tuple):
        return sorted({str(item).strip() for item in value if str(item).strip()})
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = ast.literal_eval(text)
        return normalize_tag_list(parsed)
    return sorted({item.strip() for item in text.split(",") if item.strip()})


def build_target_candidates(
    instrument_tags: Sequence[str],
    dataset_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    target_families = dataset_config.get("target_families", {})
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
        family = target_family(separation_target, target_families)
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


def target_family(
    separation_target: str,
    families: Mapping[str, Sequence[str]],
) -> str:
    lowered = separation_target.lower()
    for family_name, family_values in families.items():
        if lowered in {str(value).lower() for value in family_values}:
            return str(family_name)
    return "other"


def select_stratified(
    candidates: Sequence[dict[str, Any]],
    limit: int,
    seed: int,
    max_per_artist: int,
    max_per_album: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    relax_limits = False

    while len(selected) < limit:
        covered = coverage(selected)
        best: dict[str, Any] | None = None
        best_score: tuple[float, float] | None = None
        artist_counts = value_counts(selected, "artist_id")
        album_counts = value_counts(selected, "album_id")

        for candidate in pool:
            if candidate["track_id"] in selected_ids:
                continue
            if not relax_limits and artist_counts.get(candidate["artist_id"], 0) >= max_per_artist:
                continue
            if not relax_limits and album_counts.get(candidate["album_id"], 0) >= max_per_album:
                continue
            score = (
                selection_score(candidate, covered),
                float(candidate["_noise"]),
            )
            if best_score is None or score > best_score:
                best = candidate
                best_score = score

        if best is None:
            if not relax_limits:
                relax_limits = True
                continue
            raise ValueError("Only selected %d compatible tracks from %d candidates." % (len(selected), len(candidates)))

        selected_ids.add(str(best["track_id"]))
        updated = dict(best)
        updated["selection_rank"] = len(selected) + 1
        selected.append(updated)

    return selected


def coverage(selected: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    return {
        "families": flatten_values(selected, "target_families"),
        "stems": flatten_values(selected, "target_stems"),
        "instruments": flatten_values(selected, "instrument_tags"),
        "genres": flatten_values(selected, "genres"),
        "moods": flatten_values(selected, "mood_themes"),
    }


def flatten_values(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> set[str]:
    values: set[str] = set()
    for row in rows:
        values.update(str(value) for value in row.get(key, []))
    return values


def value_counts(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def selection_score(
    candidate: Mapping[str, Any],
    covered: Mapping[str, set[str]],
) -> float:
    return (
        100.0 * new_count(candidate["target_families"], covered["families"])
        + 35.0 * new_count(candidate["target_stems"], covered["stems"])
        + 10.0 * new_count(candidate["instrument_tags"], covered["instruments"])
        + 4.0 * new_count(candidate["genres"], covered["genres"])
        + 3.0 * new_count(candidate["mood_themes"], covered["moods"])
        + float(len(candidate["target_candidates"]))
    )


def new_count(
    values: Sequence[str],
    covered: set[str],
) -> int:
    return len({str(value) for value in values} - covered)


def write_audio_files(
    selected: Sequence[dict[str, Any]],
    output_root: Path,
    force_audio: bool,
    dataset_root: Path,
    tar_cache_dir: Path,
    download_from: str,
    tar_mode: str,
) -> None:
    tar_cache_dir.mkdir(parents=True, exist_ok=True)
    for row in tqdm(selected, desc="Writing selected audio", unit="track"):
        audio_path = output_root / str(row["relative_path"])
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        if audio_path.exists() and not force_audio:
            tqdm.write("Audio exists, skipping decode | track=%s | path=%s" % (row["track_id"], audio_path))
            continue
        tqdm.write(
            "Preparing full-quality audio | track=%s | source=%s | tar=%s"
            % (row["track_id"], row["source_path"], row["_tar_name"])
        )
        source_mp3 = extract_full_quality_source(
            row=row,
            dataset_root=dataset_root,
            tar_cache_dir=tar_cache_dir,
            download_from=download_from,
            tar_mode=tar_mode,
        )
        tqdm.write("Decoding source MP3 to WAV | track=%s | source=%s | output=%s" % (row["track_id"], source_mp3, audio_path))
        write_decoded_audio(source_mp3, audio_path)
        tqdm.write("Wrote decoded WAV | track=%s | output=%s" % (row["track_id"], audio_path))


def extract_full_quality_source(
    row: Mapping[str, Any],
    dataset_root: Path,
    tar_cache_dir: Path,
    download_from: str,
    tar_mode: str,
) -> Path:
    source_path = str(row["source_path"])
    tar_name = str(row["_tar_name"])
    permanent_source_path = tar_cache_dir / "tracks" / source_path
    if permanent_source_path.exists():
        expected_sha256 = row.get("source_sha256")
        if expected_sha256 in (None, "") or compute_sha256(permanent_source_path) == expected_sha256:
            tqdm.write("Using cached extracted MP3 | source=%s" % source_path)
            return permanent_source_path
        permanent_source_path.unlink()
    if tar_mode == "stream":
        return stream_extract_full_quality_source(
            row=row,
            tar_cache_dir=tar_cache_dir,
            download_from=download_from,
        )
    tar_path = tar_cache_dir / tar_name
    tqdm.write("Checking full-quality tar cache | tar=%s | path=%s" % (tar_name, tar_path))
    if not valid_full_quality_tar(
        dataset_root=dataset_root,
        tar_name=tar_name,
        tar_path=tar_path,
    ):
        if tar_path.exists():
            tqdm.write("Cached tar failed checksum, deleting | tar=%s | path=%s" % (tar_name, tar_path))
            tar_path.unlink()
        tqdm.write("Downloading full-quality tar | tar=%s" % tar_name)
        download_full_quality_tar(
            dataset_root=dataset_root,
            tar_name=tar_name,
            output_path=tar_path,
            download_from=download_from,
        )
    else:
        tqdm.write("Using validated cached tar | tar=%s" % tar_name)
    tqdm.write("Extracting selected MP3 from tar | tar=%s | member_suffix=%s" % (tar_name, source_path))
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        with tarfile.open(tar_path, "r") as archive:
            member_name = tar_member_name(archive, source_path)
            archive.extract(member_name, path=temp_path)
        extracted_path = temp_path / member_name
        permanent_source_path.parent.mkdir(parents=True, exist_ok=True)
        permanent_source_path.write_bytes(extracted_path.read_bytes())
    expected_sha256 = row.get("source_sha256")
    tqdm.write("Validating extracted MP3 checksum | source=%s" % source_path)
    if expected_sha256 not in (None, "") and compute_sha256(permanent_source_path) != expected_sha256:
        raise ValueError("Checksum mismatch for full-quality MTG audio %s." % source_path)
    return permanent_source_path


def stream_extract_full_quality_source(
    row: Mapping[str, Any],
    tar_cache_dir: Path,
    download_from: str,
) -> Path:
    source_path = str(row["source_path"])
    tar_name = str(row["_tar_name"])
    permanent_source_path = tar_cache_dir / "tracks" / source_path
    permanent_source_path.parent.mkdir(parents=True, exist_ok=True)
    if download_from == "mtg":
        base_url = "https://essentia.upf.edu/documentation/datasets/mtg-jamendo"
    else:
        base_url = "https://cdn.freesound.org/mtg-jamendo"
    url = "%s/raw_30s/audio/%s" % (base_url, tar_name)
    tqdm.write("Streaming full-quality tar until target member | tar=%s | target=%s" % (tar_name, source_path))
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    try:
        with tarfile.open(fileobj=response.raw, mode="r|*") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                if not member.name.endswith("/%s" % source_path):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError("Could not read streamed tar member %s." % member.name)
                temp_path = permanent_source_path.with_suffix(permanent_source_path.suffix + ".part")
                with temp_path.open("wb") as handle:
                    for chunk in iter(lambda: extracted.read(512 * 1024), b""):
                        handle.write(chunk)
                temp_path.replace(permanent_source_path)
                expected_sha256 = row.get("source_sha256")
                tqdm.write("Validating streamed MP3 checksum | source=%s" % source_path)
                if expected_sha256 not in (None, "") and compute_sha256(permanent_source_path) != expected_sha256:
                    permanent_source_path.unlink()
                    raise ValueError("Checksum mismatch for full-quality MTG audio %s." % source_path)
                return permanent_source_path
    finally:
        response.close()
    raise ValueError("Could not find %s while streaming %s." % (source_path, tar_name))


def download_full_quality_tar(
    dataset_root: Path,
    tar_name: str,
    output_path: Path,
    download_from: str,
) -> None:
    sha256_tars = load_track_sha256(dataset_root / "data" / "download" / "raw_30s_audio_sha256_tars.txt")
    if download_from == "mtg":
        base_url = "https://essentia.upf.edu/documentation/datasets/mtg-jamendo"
    else:
        base_url = "https://cdn.freesound.org/mtg-jamendo"
    url = "%s/raw_30s/audio/%s" % (base_url, tar_name)
    tqdm.write("Downloading tar from %s" % url)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    total = response.headers.get("Content-Length")
    total_bytes = int(total) if total is not None else None
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    with temp_path.open("wb") as handle:
        with tqdm(total=total_bytes, unit="B", unit_scale=True, desc="Downloading %s" % tar_name) as progress:
            for chunk in response.iter_content(chunk_size=512 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    expected_sha256 = sha256_tars.get(tar_name)
    tqdm.write("Validating downloaded tar checksum | tar=%s" % tar_name)
    if expected_sha256 not in (None, "") and compute_sha256(temp_path) != expected_sha256:
        temp_path.unlink()
        raise ValueError("Checksum mismatch for full-quality MTG tar %s." % tar_name)
    temp_path.replace(output_path)
    tqdm.write("Cached full-quality tar | tar=%s | path=%s" % (tar_name, output_path))


def valid_full_quality_tar(
    dataset_root: Path,
    tar_name: str,
    tar_path: Path,
) -> bool:
    if not tar_path.exists():
        tqdm.write("Tar cache miss | tar=%s" % tar_name)
        return False
    sha256_tars = load_track_sha256(dataset_root / "data" / "download" / "raw_30s_audio_sha256_tars.txt")
    expected_sha256 = sha256_tars.get(tar_name)
    if expected_sha256 in (None, ""):
        tqdm.write("No tar checksum listed, trusting existing tar | tar=%s" % tar_name)
        return True
    tqdm.write("Validating cached tar checksum | tar=%s" % tar_name)
    valid = compute_sha256(tar_path) == expected_sha256
    tqdm.write("Cached tar checksum %s | tar=%s" % ("OK" if valid else "FAILED", tar_name))
    return valid


def tar_member_name(archive: tarfile.TarFile, source_path: str) -> str:
    suffix = "/%s" % source_path
    for member in archive.getmembers():
        if member.isfile() and member.name.endswith(suffix):
            return member.name
    raise ValueError("Could not find %s in %s." % (source_path, archive.name))


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_decoded_audio(
    source_path: Path,
    audio_path: Path,
) -> None:
    audio_data, sample_rate = soundfile.read(
        source_path,
        always_2d=True,
        dtype="float32",
    )
    soundfile.write(audio_path, audio_data, sample_rate)


def write_filtered_metadata(
    selected: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> Path:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = metadata_dir / "mtg_filtered_demucs6s_2sources.tsv"
    with metadata_path.open("w", encoding="utf-8") as handle:
        handle.write("TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n")
        for row in selected:
            tags = prefixed_tags("genre", row["genres"])
            tags.extend(prefixed_tags("instrument", row["instrument_tags"]))
            tags.extend(prefixed_tags("mood/theme", row["mood_themes"]))
            fields = [
                str(row["track_id"]),
                str(row["artist_id"]),
                str(row["album_id"]),
                str(row["relative_path"]),
                "%.1f" % float(row["duration_s"]),
            ]
            fields.extend(tags)
            handle.write("\t".join(fields))
            handle.write("\n")
    return metadata_path


def prefixed_tags(
    prefix: str,
    values: Sequence[str],
) -> list[str]:
    return ["%s---%s" % (prefix, value) for value in values]


def selected_track_rows(selected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected:
        kept = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        rows.append(kept)
    return rows


def write_all_plans(
    analysis_rows: Sequence[Mapping[str, Any]],
    plans_path: Path,
    config_dir: Path,
    max_variants_per_recipe: int | None,
    variant_selection: str,
    random_plans_per_clip: int,
    seed: int,
) -> int:
    planner = GroundTruthPlanner.from_directory(config_dir)
    plans_path.parent.mkdir(parents=True, exist_ok=True)
    plan_count = 0
    with plans_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(analysis_rows, desc="Generating all validation plans", unit="track"):
            plans = planner.plan(
                metadata=record,
                mode="enumerate",
                max_variants_per_recipe=max_variants_per_recipe,
                variant_selection=variant_selection,
                seed=stable_seed(record.get("clip_id"), seed),
            )
            plans.extend(
                planner.random_plans(
                    metadata=record,
                    count=random_plans_per_clip,
                    seed=stable_seed(record.get("clip_id"), "random", seed),
                )
            )
            for plan in plans:
                handle.write(json.dumps(plan_row(record, plan, planner), sort_keys=True))
                handle.write("\n")
                plan_count += 1
    return plan_count


def plan_row(
    record: Mapping[str, Any],
    plan: Any,
    planner: GroundTruthPlanner,
) -> dict[str, Any]:
    row = plan.to_dict()
    analysis = dict(record.get("analysis", {}))
    row["clip_id"] = record.get("clip_id")
    row["audio_path"] = record.get("audio_path")
    row["genres"] = list(analysis.get("genres", []))
    row["mood_themes"] = list(analysis.get("mood_themes", []))
    row["issues"] = list(analysis.get("issues", []))
    row["target_stem"] = plan.bindings.get("target_description")
    row["target_family"] = plan.bindings.get("target_family")
    target_candidate = plan.bindings.get("target_candidate") or {}
    row["separation_target"] = target_candidate.get("separation_target")
    row["graph_description"] = planner.describe_plan(plan, validate=False)
    return row


def build_summary(
    selected: Sequence[Mapping[str, Any]],
    output_root: Path,
    selected_tracks_path: Path,
    metadata_path: Path,
    analysis_path: Path,
    plans_path: Path,
    plan_count: int,
) -> dict[str, Any]:
    return {
        "output_root": str(output_root),
        "num_tracks": len(selected),
        "num_plans": plan_count,
        "selected_tracks_path": str(selected_tracks_path),
        "metadata_path": str(metadata_path),
        "analysis_path": str(analysis_path),
        "plans_path": str(plans_path),
        "demucs_cache_dir": str(output_root / "demucs_cache"),
        "render_output_root": str(output_root / "renders"),
        "selected_track_ids": [str(row["track_id"]) for row in selected],
        "target_family_counts": sorted_counts(selected, "target_families"),
        "target_stem_counts": sorted_counts(selected, "target_stems"),
        "genre_counts": sorted_counts(selected, "genres"),
        "mood_theme_counts": sorted_counts(selected, "mood_themes"),
    }


def sorted_counts(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[list[Any]]:
    counts: dict[str, int] = {}
    for row in rows:
        for value in row.get(key, []):
            text = str(value)
            counts[text] = counts.get(text, 0) + 1
    return [[key, counts[key]] for key in sorted(counts, key=lambda item: (-counts[item], item))]


def stable_noise(
    seed: int,
    track_id: int,
) -> float:
    digest = hashlib.sha256(("%d:%d" % (seed, track_id)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def stable_seed(*parts: object) -> int:
    joined = "||".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


if __name__ == "__main__":
    main()
