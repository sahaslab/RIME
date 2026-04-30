import sys
import argparse
import csv
import hashlib
import json
import subprocess
import soundfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-tracks-path", type=Path, default=Path("derived/validation/mtg_jamendo_10/selected_tracks.jsonl"), help="Selected tracks JSONL from validation subset prep. Default: %(default)s")
    parser.add_argument("--metadata-path", type=Path, default=Path("mtg-jamendo-dataset/data/raw_30s_cleantags.tsv"), help="Local MTG-Jamendo metadata TSV. Default: %(default)s")
    parser.add_argument("--sha256-path", type=Path, default=Path("mtg-jamendo-dataset/data/download/raw_30s_audio_sha256_tracks.txt"), help="Full-quality MP3 checksum manifest. Default: %(default)s")
    parser.add_argument("--cluster-root", type=str, default="discovery8:/dartfs-hpc/rc/home/f/f007krf/lab/shared/MTG-Jamendo", help="Remote cluster root containing <prefix>/<id>.mp3. Default: %(default)s")
    parser.add_argument("--mp3-cache-dir", type=Path, default=Path("derived/validation/mtg_jamendo_10/mp3_cache"), help="Local cache for rsynced MP3s. Default: %(default)s")
    parser.add_argument("--audio-root", type=Path, default=Path("derived/validation/mtg_jamendo_10/audio"), help="Output decoded WAV root. Default: %(default)s")
    parser.add_argument("--max-audio-seconds", type=float, default=0.0, help="Decoded WAV duration cap. Use 0 for full track. Default: %(default)s")
    parser.add_argument("--force", action="store_true", help="Overwrite existing MP3 cache and WAV outputs. Default: reuse")
    args = parser.parse_args()

    selected_ids = selected_track_ids(args.selected_tracks_path)
    metadata = metadata_by_track_id(args.metadata_path)
    checksums = load_track_sha256(args.sha256_path)
    args.mp3_cache_dir.mkdir(parents=True, exist_ok=True)
    args.audio_root.mkdir(parents=True, exist_ok=True)

    for track_id in selected_ids:
        row = metadata[track_id]
        source_path = str(row["PATH"])
        mp3_path = args.mp3_cache_dir / source_path
        wav_path = args.audio_root / ("%s.wav" % track_id)
        if not mp3_path.exists() or args.force:
            rsync_one(
                remote_path="%s/%s" % (args.cluster_root.rstrip("/"), source_path),
                local_path=mp3_path,
            )
        expected_sha256 = checksums.get(source_path)
        if expected_sha256 not in (None, "") and compute_sha256(mp3_path) != expected_sha256:
            raise ValueError("Checksum mismatch for %s at %s." % (track_id, mp3_path))
        if not wav_path.exists() or args.force:
            decode_mp3_to_wav(
                mp3_path,
                wav_path,
                max_audio_seconds=args.max_audio_seconds,
            )
        print("ready %s | mp3=%s | wav=%s | max_audio_seconds=%s" % (track_id, mp3_path, wav_path, args.max_audio_seconds))


def selected_track_ids(path: Path) -> list[str]:
    ids = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(str(json.loads(line)["track_id"]))
    return ids


def metadata_by_track_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t", restkey="EXTRA_TAGS")
        return {str(row["TRACK_ID"]): dict(row) for row in reader}


def load_track_sha256(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            pieces = line.strip().split()
            if len(pieces) >= 2:
                checksums[pieces[1]] = pieces[0]
    return checksums


def rsync_one(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "rsync",
        "-avP",
        remote_path,
        str(local_path),
    ]
    print(" ".join(command))
    subprocess.run(command, check=True)


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_mp3_to_wav(
    mp3_path: Path,
    wav_path: Path,
    max_audio_seconds: float,
) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio_data, sample_rate = soundfile.read(
        mp3_path,
        always_2d=True,
        dtype="float32",
    )
    if max_audio_seconds > 0.0:
        max_samples = int(max_audio_seconds * sample_rate)
        if max_samples > 0:
            audio_data = audio_data[:max_samples]
    soundfile.write(wav_path, audio_data, sample_rate)


if __name__ == "__main__":
    main()
