import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_truth.datasets import available_dataset_names, build_analysis_manifest
from ground_truth.io_utils import write_jsonl


def main():
    parser = argparse.ArgumentParser()
    dataset_names = available_dataset_names()
    parser.add_argument("--dataset-name", choices=dataset_names, default="mtg_jamendo", help="Dataset adapter to use. Default: %(default)s")
    parser.add_argument("--metadata-dir", type=Path, default=None, help="Directory containing dataset metadata tables. Default depends on the dataset.")
    parser.add_argument("--dataset-config", type=Path, default=None, help="Dataset adapter config. Default depends on the dataset.")
    parser.add_argument("--audio-root", type=Path, default=None, help="Optional root directory containing audio files matching the dataset relative paths. Default depends on the dataset.")
    parser.add_argument("--output-path", type=Path, default=None, help="Output JSONL path for the analysis manifest. Default: ~/lab/postmaster/ground_truth/<dataset>_analysis_manifest.jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Optional clip limit for smoke tests or partial builds. Default: no limit")
    parser.add_argument(
        "--separation-profile",
        choices=["demucs_6s", "none"],
        default="demucs_6s",
        help="Target filtering profile used to keep only separation-compatible candidates. Default: %(default)s",
    )
    args = parser.parse_args()

    if args.dataset_config is None:
        args.dataset_config = Path(f"configs/ground_truth/datasets/{args.dataset_name}.yaml")
    if args.metadata_dir is None:
        if args.dataset_name == "mtg_jamendo":
            args.metadata_dir = Path("~/lab/shared/mtg-jamendo-dataset/data").expanduser()
        elif args.dataset_name == "medleydb":
            args.metadata_dir = Path("~/lab/shared/medleydb").expanduser()
    if args.audio_root is None:
        if args.dataset_name == "mtg_jamendo":
            args.audio_root = Path("~/lab/shared/MTG-Jamendo").expanduser()
    if args.output_path is None:
        args.output_path = Path(f"~/lab/postmaster/ground_truth/{args.dataset_name}_analysis_manifest.jsonl").expanduser()

    rows = build_analysis_manifest(
        dataset_name=args.dataset_name,
        metadata_dir=args.metadata_dir,
        dataset_config_path=args.dataset_config,
        audio_root=args.audio_root,
        limit=args.limit,
        separation_profile=None if args.separation_profile == "none" else args.separation_profile,
    )
    write_jsonl(args.output_path, rows)


if __name__ == "__main__":
    main()
