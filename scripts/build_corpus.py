import json
import hashlib
import math
import re
import argparse
import shutil
import urllib.request
from pathlib import Path
from datetime import datetime, timezone
from typing import Any
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        epilog="Output: 01_corpus/original_yaml, inventory.json and source assets. Start with --base-config-dir /path/to/RIME/configs/ground_truth; reruns can use the bundled originals. Python 3.11 setup: pip install uv==0.6.14; uv pip install numpy==2.2.4 pandas==2.2.3 pyarrow==19.0.1 PyYAML==6.0.2 scipy==1.14.1 scikit-learn==1.6.1 joblib==1.4.2 threadpoolctl==3.6.0 tqdm==4.67.1 torch==2.6.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "01_corpus",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Verify existing assets without downloading",
    )
    parser.add_argument(
        "--asset-cache",
        type=Path,
        default=None,
        help="Reuse matching files from an existing corpus",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
    )
    parser.add_argument(
        "--base-config-dir",
        type=Path,
        default=ROOT.parent / "configs" / "prior_baseline",
        help="Original RIME config directory; defaults to the bundled originals",
    )
    args = parser.parse_args()
    original_dir = args.output_dir / "original_yaml"
    originals = {
        name: (args.base_config_dir / (name + ".yaml")).read_bytes()
        for name in ["distributions", "motifs", "recipes", "operators", "constraints"]
    }
    if original_dir.exists() and any(
        (original_dir / (name + ".yaml")).read_bytes() != data for name, data in originals.items()
    ):
        backup_existing(original_dir)
    original_dir.mkdir(parents=True, exist_ok=True)
    for name, data in originals.items():
        (original_dir / (name + ".yaml")).write_bytes(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, asset in enumerate(tqdm(ASSETS, desc="Acquiring corpus"), start=1):
        path = args.output_dir / asset["path"]
        if path.exists():
            assert digest(path) == asset["sha256"], "Asset changed: %s" % path
            continue
        print("[%d/%d] %s" % (index, len(ASSETS), asset["path"]), flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        if args.asset_cache is not None:
            candidates = [
                args.asset_cache / asset["path"],
                args.asset_cache / path.name,
            ]
            cached = next(
                (
                    item
                    for item in candidates
                    if item.is_file() and digest(item) == asset["sha256"]
                ),
                None,
            )
            if cached is not None:
                shutil.copyfile(cached, path)
                continue
        assert not args.offline, "Missing asset: %s" % path
        temporary = path.with_suffix(path.suffix + ".download")
        request = urllib.request.Request(
            asset["url"],
            headers={"User-Agent": "rime-priors"},
        )
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        assert digest(temporary) == asset["sha256"], "Downloaded bytes changed: %s" % path
        temporary.replace(path)
    write_json(
        args.output_dir / "inventory.json",
        {"sources": SOURCES, "assets": ASSETS},
    )
    print(
        "01_corpus: verified %d assets; inventory.json records their URLs and hashes"
        % len(ASSETS),
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stable_id(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        allow_nan=False,
    ).encode(
        "utf-8",
    )
    return hashlib.sha256(encoded).hexdigest()[:24]


def numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    match = re.fullmatch(
        r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*(?::\s*1)?\s*",
        str(value),
    )
    if match is None:
        return None
    result = float(match.group(1))
    return result if math.isfinite(result) else None


def flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in flatten(child, "%s/%s" % (prefix, key))
        ]
    if isinstance(value, list):
        return [
            item
            for index, child in enumerate(value)
            for item in flatten(child, "%s/%d" % (prefix, index))
        ]
    return [(prefix.lstrip("/"), value)]


def backup_existing(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    suffix = path.suffix if path.is_file() else ""
    backup = path.parent / (path.name + ".backups") / (stamp + suffix)
    backup.parent.mkdir(parents=True, exist_ok=True)
    path.rename(backup)
    return backup


ROOT = Path(__file__).resolve().parents[1] / "generation_priors"
PIPELINE_SCRIPTS = [
    Path(__file__).resolve().parent / name
    for name in ("build_corpus.py", "extract_priors.py", "export_priors_to_yaml.py")
]

# Pinned source metadata and asset hashes.
SOURCES = {
    "calf": {
        "adapter": "calf",
        "evidence": "Factory presets; include instruments in native archive",
        "include": [
            "presets.xml",
            "COPYING*",
            "LICENSE*",
            "README*",
            "src/metadata.cpp",
        ],
        "kind": "github",
        "license": "Consult locked COPYING and per-file terms",
        "repository": "calf-studio-gear/calf",
        "revision": "58d033bc6210c3b1edb01e890d9434f989daa13b",
    },
    "dafx_2017": {
        "adapter": "paper",
        "evidence": "Six reported chain percentages; incomplete support over 178 submissions",
        "files": {
            "paper.pdf": "https://www.dafx.de/paper-archive/2017/papers/DAFx17_paper_75.pdf",
        },
        "kind": "urls",
        "license": "Paper copyright; published aggregate statistics",
    },
    "easyeffects_digitalone": {
        "adapter": "easyeffects",
        "evidence": "Playback presets; explicit plugin order; not production prevalence",
        "include": [
            "*.json",
            "*.irs",
            "*.wav",
            "*.sofa",
            "LICENSE*",
            "COPYING*",
            "README*",
            "*.md",
        ],
        "kind": "github",
        "license": "Consult locked license and per-file terms",
        "repository": "Digitalone1/EasyEffects-Presets",
        "revision": "6fc0630f3d18f5668b11ebe4846179914b0bd24e",
    },
    "easyeffects_jackhack": {
        "adapter": "easyeffects",
        "evidence": "Playback presets; explicit plugin order; not production prevalence",
        "include": [
            "*.json",
            "*.irs",
            "*.wav",
            "*.sofa",
            "LICENSE*",
            "COPYING*",
            "README*",
            "*.md",
        ],
        "kind": "github",
        "license": "MIT repository; inspect separate impulse-response terms",
        "repository": "JackHack96/EasyEffects-Presets",
        "revision": "dd966e41ad9e44d4b11e19047f526ba718bbbe57",
    },
    "easyeffects_schema": {
        "adapter": "documentation",
        "evidence": "Control units and preset serialization for translation review",
        "include": [
            "src/contents/ui/Compressor.qml",
            "src/compressor.cpp",
            "src/compressor_preset.cpp",
            "LICENSE*",
            "COPYING*",
        ],
        "kind": "github",
        "license": "GPL-3.0; consult locked license",
        "repository": "wwmm/easyeffects",
        "revision": "4103f81d4a161b8a93a7cefebd965752ea529990",
    },
    "mixassist": {
        "adapter": "mixassist",
        "evidence": "Dialogue language; no logged processor settings or chain orders",
        "files": [
            "README.md",
            "data/train-00000-of-00001.parquet",
            "data/validation-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ],
        "kind": "huggingface",
        "license": "No license declared in inspected dataset card",
        "repository": "mclemcrew/MixAssist",
        "revision": "bf37575232cff9490085433fe1434e23f5426b52",
    },
    "mixparams": {
        "adapter": "mixparams",
        "evidence": "Annotated channel state; plugin units vary; insert order unavailable",
        "files": [
            "README.md",
            "data/train-00000-of-00001.parquet",
            "data/dev-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ],
        "kind": "huggingface",
        "license": "No license declared in inspected dataset card",
        "repository": "mclemcrew/MixParams",
        "revision": "df15a107cf86ed46793df859daf43d5f24b142fd",
    },
    "socialfx": {
        "adapter": "socialfx",
        "evidence": "Crowd descriptor-control pairs; EQ values are relative spectral curves",
        "files": [
            "README.md",
            "data/comp-00000-of-00001.parquet",
            "data/eq-00000-of-00001.parquet",
            "data/reverb-00000-of-00001.parquet",
        ],
        "kind": "huggingface",
        "license": "No license declared in inspected mirror card",
        "repository": "seungheondoh/socialfx-original",
        "revision": "d7e266cac1526ae7db52ea16c744595a05b9df27",
    },
    "timbral_hierarchy": {
        "adapter": "timbral",
        "evidence": "Timbral vocabulary and search frequencies; no processor settings",
        "files": {
            "TimbralHierarchy.zip": "https://zenodo.org/records/167392/files/TimbralHierarchy.zip?download=1",
        },
        "kind": "urls",
        "license": "Metadata CC BY 4.0; bundled Contents.txt states noncommercial terms",
        "record": 167392,
        "sha256": {
            "TimbralHierarchy.zip": "5c0a8ad997dfa53daba92690a97ff8f10bb69c9f84354243eb3deb9adcec6327",
        },
    },
}

ASSETS = [
    {
        "source": "calf",
        "path": "calf/COPYING",
        "url": "https://raw.githubusercontent.com/calf-studio-gear/calf/58d033bc6210c3b1edb01e890d9434f989daa13b/COPYING",
        "sha256": "512d2d21b6b3384ba64781abb0208a1b87740bc31e2df48e2b206ddb7e4d5779",
    },
    {
        "source": "calf",
        "path": "calf/COPYING.ASSETS",
        "url": "https://raw.githubusercontent.com/calf-studio-gear/calf/58d033bc6210c3b1edb01e890d9434f989daa13b/COPYING.ASSETS",
        "sha256": "e7879d0de9d296d23f9695c1d8b76e92834640eb030768acb1debc4cda9ae035",
    },
    {
        "source": "calf",
        "path": "calf/COPYING.GPL",
        "url": "https://raw.githubusercontent.com/calf-studio-gear/calf/58d033bc6210c3b1edb01e890d9434f989daa13b/COPYING.GPL",
        "sha256": "32b1062f7da84967e7019d01ab805935caa7ab7321a7ced0e30ebe75e5df1670",
    },
    {
        "source": "calf",
        "path": "calf/README.md",
        "url": "https://raw.githubusercontent.com/calf-studio-gear/calf/58d033bc6210c3b1edb01e890d9434f989daa13b/README.md",
        "sha256": "8701e0a333332b91c64298063abc04155fda5b5a6731ed07efdc3aa8963f9527",
    },
    {
        "source": "calf",
        "path": "calf/presets.xml",
        "url": "https://raw.githubusercontent.com/calf-studio-gear/calf/58d033bc6210c3b1edb01e890d9434f989daa13b/presets.xml",
        "sha256": "fc2b3d732f24624714c669a71f3c30bbaba7e85caafa4355d4df5437a57435cf",
    },
    {
        "source": "calf",
        "path": "calf/src/metadata.cpp",
        "url": "https://raw.githubusercontent.com/calf-studio-gear/calf/58d033bc6210c3b1edb01e890d9434f989daa13b/src/metadata.cpp",
        "sha256": "4da918b45421d13730419d726f2c36c58efc315bec04d0a6a3e0ad750ee70b88",
    },
    {
        "source": "dafx_2017",
        "path": "dafx_2017/paper.pdf",
        "url": "https://www.dafx.de/paper-archive/2017/papers/DAFx17_paper_75.pdf",
        "sha256": "78e5c33e4f0468ab264ee417d9c254782d7c73ae74058b3360dd0eb4882a135c",
    },
    {
        "source": "easyeffects_digitalone",
        "path": "easyeffects_digitalone/LICENSE",
        "url": "https://raw.githubusercontent.com/Digitalone1/EasyEffects-Presets/6fc0630f3d18f5668b11ebe4846179914b0bd24e/LICENSE",
        "sha256": "46787cc7f156dad1de0259ff03c3da03e2ef2859f6f07bb6ee00947c227693fd",
    },
    {
        "source": "easyeffects_digitalone",
        "path": "easyeffects_digitalone/LoudnessCrystalEqualizer-GTK.json",
        "url": "https://raw.githubusercontent.com/Digitalone1/EasyEffects-Presets/6fc0630f3d18f5668b11ebe4846179914b0bd24e/LoudnessCrystalEqualizer-GTK.json",
        "sha256": "716ccf48833c0b7da92a958289ee8b57485921284413f554df1faee347360c55",
    },
    {
        "source": "easyeffects_digitalone",
        "path": "easyeffects_digitalone/LoudnessCrystalEqualizer.json",
        "url": "https://raw.githubusercontent.com/Digitalone1/EasyEffects-Presets/6fc0630f3d18f5668b11ebe4846179914b0bd24e/LoudnessCrystalEqualizer.json",
        "sha256": "4c8d5fe4a41e9ac1e62265c618b18c742544858bc40b56578da2cd88e8651508",
    },
    {
        "source": "easyeffects_digitalone",
        "path": "easyeffects_digitalone/LoudnessEqualizer-GTK.json",
        "url": "https://raw.githubusercontent.com/Digitalone1/EasyEffects-Presets/6fc0630f3d18f5668b11ebe4846179914b0bd24e/LoudnessEqualizer-GTK.json",
        "sha256": "969867132b917acb584c4b6ca47864a5d1b49f6a0acca7eb5f92f55fbc190789",
    },
    {
        "source": "easyeffects_digitalone",
        "path": "easyeffects_digitalone/LoudnessEqualizer-OldGate.json",
        "url": "https://raw.githubusercontent.com/Digitalone1/EasyEffects-Presets/6fc0630f3d18f5668b11ebe4846179914b0bd24e/LoudnessEqualizer-OldGate.json",
        "sha256": "c89573abf1246899c03c94794a8982894e30b79996cccf2e2bf5d66b75e16043",
    },
    {
        "source": "easyeffects_digitalone",
        "path": "easyeffects_digitalone/LoudnessEqualizer-PE.json",
        "url": "https://raw.githubusercontent.com/Digitalone1/EasyEffects-Presets/6fc0630f3d18f5668b11ebe4846179914b0bd24e/LoudnessEqualizer-PE.json",
        "sha256": "0971652c7842a118871347ef74e45b1c196d24894a7a123b50e1a357bfc15a2c",
    },
    {
        "source": "easyeffects_digitalone",
        "path": "easyeffects_digitalone/LoudnessEqualizer.json",
        "url": "https://raw.githubusercontent.com/Digitalone1/EasyEffects-Presets/6fc0630f3d18f5668b11ebe4846179914b0bd24e/LoudnessEqualizer.json",
        "sha256": "99d60fee1a146d93c78c5500b0220c0f58d42a832856404e6e90b4d248a778f4",
    },
    {
        "source": "easyeffects_digitalone",
        "path": "easyeffects_digitalone/README.md",
        "url": "https://raw.githubusercontent.com/Digitalone1/EasyEffects-Presets/6fc0630f3d18f5668b11ebe4846179914b0bd24e/README.md",
        "sha256": "78dbfc09d1f153bfa5ffd187733f02efa8f6081f3ac4ce6b8a88ee7a9225956f",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/Advanced Auto Gain.json",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/Advanced%20Auto%20Gain.json",
        "sha256": "017cf2d3839131e837f47ee88e446dbae993d075349ca2e453548a9a60fdcf34",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/Bass Boosted.json",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/Bass%20Boosted.json",
        "sha256": "f79b40b13a043b36b66a120e375e685f0563853a03c827ea488b87423995579b",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/Bass Enhancing + Perfect EQ - Low Latency.json",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/Bass%20Enhancing%20%2B%20Perfect%20EQ%20-%20Low%20Latency.json",
        "sha256": "ba353a83b7c6cf1ca32ced5e5edb09c6173751d3380d8ca8d47cba200bdd14be",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/Bass Enhancing + Perfect EQ.json",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/Bass%20Enhancing%20%2B%20Perfect%20EQ.json",
        "sha256": "750ce30bfafefb303259a6819aacae07fd3d2ec25041fb47470fc6cf02e9d7c4",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/Boosted.json",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/Boosted.json",
        "sha256": "ebfb7ee79018baa61ac5ac895c9661cc0cac57fc7440558be1fade858f7eb3f1",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/Dolby Atmos.json",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/Dolby%20Atmos.json",
        "sha256": "e8bf1849349dd131072b4a61bd845cc8119464c6796ec2d1842f6f5173ee0edd",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/LICENSE",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/LICENSE",
        "sha256": "7085b690bded564440e10ed3922aeb98f6e1ff08a8c172d5056ac5a6da0e9553",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/Loudness+Autogain.json",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/Loudness%2BAutogain.json",
        "sha256": "332a68ee1776c34be755b97181e0f34f63048fc3de024e3027dc21a40ef20e50",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/Perfect EQ.json",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/Perfect%20EQ.json",
        "sha256": "2e15dd8fdee2160068b806dcc5f69b92c48124efc0a2e1cfbf5adccb6671f732",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/README.md",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/README.md",
        "sha256": "642b85fd366b03ff5f91afc7e04f08ddd114d17edcfa72d4bfc62187975248e0",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/Speaker Sync.json",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/Speaker%20Sync.json",
        "sha256": "1ad97cfd7d2545aadb1b33ae9e0327fdf9ab6c4765620181ef438795b6b30668",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) Earpods HIFI.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20Earpods%20HIFI.irs",
        "sha256": "fed3b027ee243387d9bf24b3dc766bb304e1367d7dcd3af765aa6e155d15b24e",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) MDR-E9LP HIFI.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20MDR-E9LP%20HIFI.irs",
        "sha256": "f2f02a8219353896c46e7fdaeccf2681aa61ea36e93f1dc149b2060488d30d65",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) MDR-E9LP SM SRH940.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20MDR-E9LP%20SM%20SRH940.irs",
        "sha256": "25728e1c59ec97eeb9efbf48e927a05cbe9805947b091b28a8d9c3bd46818af1",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) MDR-E9LP SM XBA3.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20MDR-E9LP%20SM%20XBA3.irs",
        "sha256": "a258e2835edf6ff3d618c199f69e13078846ca4013a15e4ca5f0b1d603ff5b76",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) MDR-E9LP SM beyerT1.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20MDR-E9LP%20SM%20beyerT1.irs",
        "sha256": "955a8db9d33a0abfd8d098be0f6bac9f913b0dea9b0e7cc6f591c6b6b356f1dc",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) MDR-XB500 HIFI.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20MDR-XB500%20HIFI.irs",
        "sha256": "1a7a103b1714d35378e790b96c7346ed9afa9a4bdf4a32376e84ef127fbdcbb1",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) XBA-H3 HIFI.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20XBA-H3%20HIFI.irs",
        "sha256": "2d1aca8621a2fe0d3d07d3cee0b9ceb6127f9f9c08f26f023475589285df432c",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) XBA-H3 SM SRH940.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20XBA-H3%20SM%20SRH940.irs",
        "sha256": "85edf5a8fe71d1843342880006908b875ff55cd54f835325e14a14c732c6b6c8",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) XBA-H3 SM XBA4.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20XBA-H3%20SM%20XBA4.irs",
        "sha256": "1bb193acbc13898ce36a45a4f7b812cebbcb8e614d5496919dbcbf9690b12e4a",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Accudio ((48kHz Z.E.)) XBA-H3 SM beyerT1.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Accudio%20%28%2848kHz%20Z.E.%29%29%20XBA-H3%20SM%20beyerT1.irs",
        "sha256": "152be5b67ef1ec77c30f6c3e08e677ba34ffbc445367409a8d7bf51c1af65ae3",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Creative X-Fi ((Z-Edition)) Crystalizer 10 + Expand 10.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Creative%20X-Fi%20%28%28Z-Edition%29%29%20Crystalizer%2010%20%2B%20Expand%2010.irs",
        "sha256": "123d4723cbcbb86e39c6bce210b0ece57af6ab1f97bc1bb9c867a17021d3a0d9",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Dolby ATMOS ((128K MP3)) 1.Default.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Dolby%20ATMOS%20%28%28128K%20MP3%29%29%201.Default.irs",
        "sha256": "f45b751d92c54c188645f87fc0988667d5acb4cb6f76d5febb79556372425403",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/HTC Beats Audio ((Z-Edition)).irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/HTC%20Beats%20Audio%20%28%28Z-Edition%29%29.irs",
        "sha256": "2f89fdca6290096f9aa7e1a3a9d13247dce594dbde1f3f43d68d67e8ea94e5d5",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/MaxxAudio Pro ((128K MP3)) 4.Music w MaxxSpace Low Latency.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/MaxxAudio%20Pro%20%28%28128K%20MP3%29%29%204.Music%20w%20MaxxSpace%20Low%20Latency.irs",
        "sha256": "1d65e4c78b776b92e16d404bae4339e74be6813549af9bd54cdabe6f6b1a2cfa",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/MaxxAudio Pro ((128K MP3)) 4.Music w MaxxSpace.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/MaxxAudio%20Pro%20%28%28128K%20MP3%29%29%204.Music%20w%20MaxxSpace.irs",
        "sha256": "13352183f9b0ab2199dc70887fe83d704848f32c1233e1ecf45de90f1c2258fb",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Razor Surround ((48k Z-Edition)) 1.Stereo +0 Bass Low Latency.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Razor%20Surround%20%28%2848k%20Z-Edition%29%29%201.Stereo%20%2B0%20Bass%20Low%20Latency.irs",
        "sha256": "1cd9600080c073963e462fa1a93b8f2f47d1c55cd41b28fe6fae8ee534b73b93",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Razor Surround ((48k Z-Edition)) 2.Stereo +20 bass Low Latency.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Razor%20Surround%20%28%2848k%20Z-Edition%29%29%202.Stereo%20%2B20%20bass%20Low%20Latency.irs",
        "sha256": "9a0dbba01551de1ebca4b2d9de4a6eb54e5f63e88722c2b777d14db88ac69ed4",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Razor Surround ((48k Z-Edition)) 2.Stereo +20 bass.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Razor%20Surround%20%28%2848k%20Z-Edition%29%29%202.Stereo%20%2B20%20bass.irs",
        "sha256": "e41255be8b28427695146d435dc736deb8ff3dde01fe6befbd48f45c0be1c776",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Razor Surround ((48k Z-Edition)) 3.Stereo +30 Bass Low Latency.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Razor%20Surround%20%28%2848k%20Z-Edition%29%29%203.Stereo%20%2B30%20Bass%20Low%20Latency.irs",
        "sha256": "3bc9ed050ea2ec71a1e0db01f6844c1d1d2d68a2d63b501a9979595ff5d6339c",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Razor Surround ((48k Z-Edition)) 4.Stereo +50 Bass Low Latency.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Razor%20Surround%20%28%2848k%20Z-Edition%29%29%204.Stereo%20%2B50%20Bass%20Low%20Latency.irs",
        "sha256": "7931b139f53a7b71821cfa9f20b3dfb918d00a2b472c77b195bd089454fe64de",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Razor Surround ((48k Z-Edition)) 5.Stereo +70 Bass Low Latency.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Razor%20Surround%20%28%2848k%20Z-Edition%29%29%205.Stereo%20%2B70%20Bass%20Low%20Latency.irs",
        "sha256": "7e5ef7d8f7b6715a77a7cfceddca0f4a0133fa7adcbfe5a62632c0d5b4105aba",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Razor Surround ((48k Z-Edition)) 6.Stereo +80 Bass Low Latency.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Razor%20Surround%20%28%2848k%20Z-Edition%29%29%206.Stereo%20%2B80%20Bass%20Low%20Latency.irs",
        "sha256": "b50ba540b92a324cc81bc62adfa57ec053f4273a3c2cec31a2fd1f921f7bdf58",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Razor Surround ((48k Z-Edition)) 7.Stereo +100 Bass Low Latency.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Razor%20Surround%20%28%2848k%20Z-Edition%29%29%207.Stereo%20%2B100%20Bass%20Low%20Latency.irs",
        "sha256": "36431509fcf758b980490bedfe45eb7c86ff1da0a728c2a7aa8519aa477adfed",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Waves MaxxAudio ((Z-Edition)) AudioWizard 1.Music Low Latency.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Waves%20MaxxAudio%20%28%28Z-Edition%29%29%20AudioWizard%201.Music%20Low%20Latency.irs",
        "sha256": "06f5ce0bcaef113ea5ac3f91c7368daf28716cf84955bef4caf0fa8e3087308a",
    },
    {
        "source": "easyeffects_jackhack",
        "path": "easyeffects_jackhack/irs/Waves MaxxAudio ((Z-Edition)) AudioWizard 1.Music.irs",
        "url": "https://raw.githubusercontent.com/JackHack96/EasyEffects-Presets/dd966e41ad9e44d4b11e19047f526ba718bbbe57/irs/Waves%20MaxxAudio%20%28%28Z-Edition%29%29%20AudioWizard%201.Music.irs",
        "sha256": "cb40ce4330fcb71bbae5895006455189166d750018c4493b0bfeba0e9b717f98",
    },
    {
        "source": "easyeffects_schema",
        "path": "easyeffects_schema/LICENSE",
        "url": "https://raw.githubusercontent.com/wwmm/easyeffects/4103f81d4a161b8a93a7cefebd965752ea529990/LICENSE",
        "sha256": "589ed823e9a84c56feb95ac58e7cf384626b9cbf4fda2a907bc36e103de1bad2",
    },
    {
        "source": "easyeffects_schema",
        "path": "easyeffects_schema/src/compressor.cpp",
        "url": "https://raw.githubusercontent.com/wwmm/easyeffects/4103f81d4a161b8a93a7cefebd965752ea529990/src/compressor.cpp",
        "sha256": "50c02265e5ec88769df195bb3d14b5e44f40b47a2c3b09778e90dd093d2a5498",
    },
    {
        "source": "easyeffects_schema",
        "path": "easyeffects_schema/src/compressor_preset.cpp",
        "url": "https://raw.githubusercontent.com/wwmm/easyeffects/4103f81d4a161b8a93a7cefebd965752ea529990/src/compressor_preset.cpp",
        "sha256": "2a70b773eadb8944b3625b2fed742f7bcc8b9759d6531e650ead082b4742b958",
    },
    {
        "source": "easyeffects_schema",
        "path": "easyeffects_schema/src/contents/ui/Compressor.qml",
        "url": "https://raw.githubusercontent.com/wwmm/easyeffects/4103f81d4a161b8a93a7cefebd965752ea529990/src/contents/ui/Compressor.qml",
        "sha256": "2aec8e38af639fb4268f4bd382d9217e78b4a1463102a3dcc68936c656fa4dbc",
    },
    {
        "source": "mixassist",
        "path": "mixassist/README.md",
        "url": "https://huggingface.co/datasets/mclemcrew/MixAssist/resolve/bf37575232cff9490085433fe1434e23f5426b52/README.md",
        "sha256": "95c3c38928b3849e085ded237909ba8067d5ece423175152be8c5847448b94db",
    },
    {
        "source": "mixassist",
        "path": "mixassist/data/test-00000-of-00001.parquet",
        "url": "https://huggingface.co/datasets/mclemcrew/MixAssist/resolve/bf37575232cff9490085433fe1434e23f5426b52/data/test-00000-of-00001.parquet",
        "sha256": "b6d354c04d56ddb33fd05bf74bf2af446dcd1170c89d7509b55d777f2bed4739",
    },
    {
        "source": "mixassist",
        "path": "mixassist/data/train-00000-of-00001.parquet",
        "url": "https://huggingface.co/datasets/mclemcrew/MixAssist/resolve/bf37575232cff9490085433fe1434e23f5426b52/data/train-00000-of-00001.parquet",
        "sha256": "0d3047dea8213be7f3122bef79686d6f417958831eb65472f3b777ad1efee5ed",
    },
    {
        "source": "mixassist",
        "path": "mixassist/data/validation-00000-of-00001.parquet",
        "url": "https://huggingface.co/datasets/mclemcrew/MixAssist/resolve/bf37575232cff9490085433fe1434e23f5426b52/data/validation-00000-of-00001.parquet",
        "sha256": "5f2d830a7b016503653ffedbf49a8ab89eaee971f7149b424706b4f52bf6d187",
    },
    {
        "source": "mixparams",
        "path": "mixparams/README.md",
        "url": "https://huggingface.co/datasets/mclemcrew/MixParams/resolve/df15a107cf86ed46793df859daf43d5f24b142fd/README.md",
        "sha256": "deadb57ae558885490426e40b0082fd490c30db4ac91b583d2d9115cd772bd4a",
    },
    {
        "source": "mixparams",
        "path": "mixparams/data/dev-00000-of-00001.parquet",
        "url": "https://huggingface.co/datasets/mclemcrew/MixParams/resolve/df15a107cf86ed46793df859daf43d5f24b142fd/data/dev-00000-of-00001.parquet",
        "sha256": "28fa2d2f21c6b247fcda8034c83bbd1ccea2fde5c2a637ae06f36ea8a96a07e9",
    },
    {
        "source": "mixparams",
        "path": "mixparams/data/test-00000-of-00001.parquet",
        "url": "https://huggingface.co/datasets/mclemcrew/MixParams/resolve/df15a107cf86ed46793df859daf43d5f24b142fd/data/test-00000-of-00001.parquet",
        "sha256": "5ba3b784794f3a3f10480d02336f2dd7fcf3c23d11b7ec0f2224fe33aed7fc8f",
    },
    {
        "source": "mixparams",
        "path": "mixparams/data/train-00000-of-00001.parquet",
        "url": "https://huggingface.co/datasets/mclemcrew/MixParams/resolve/df15a107cf86ed46793df859daf43d5f24b142fd/data/train-00000-of-00001.parquet",
        "sha256": "9bc183a7f6c36ca73d83a25b2fbf472101d3e00322960b301b61a0e83676b356",
    },
    {
        "source": "socialfx",
        "path": "socialfx/README.md",
        "url": "https://huggingface.co/datasets/seungheondoh/socialfx-original/resolve/d7e266cac1526ae7db52ea16c744595a05b9df27/README.md",
        "sha256": "88c16843acbb6865d47fa7484318b2e6959772b749aeb0b4bdc6ea87db788756",
    },
    {
        "source": "socialfx",
        "path": "socialfx/data/comp-00000-of-00001.parquet",
        "url": "https://huggingface.co/datasets/seungheondoh/socialfx-original/resolve/d7e266cac1526ae7db52ea16c744595a05b9df27/data/comp-00000-of-00001.parquet",
        "sha256": "ae865437514d20ba5058c149025622c517693eb4abaac3b6e4a8069645486eba",
    },
    {
        "source": "socialfx",
        "path": "socialfx/data/eq-00000-of-00001.parquet",
        "url": "https://huggingface.co/datasets/seungheondoh/socialfx-original/resolve/d7e266cac1526ae7db52ea16c744595a05b9df27/data/eq-00000-of-00001.parquet",
        "sha256": "43802a449efa8445285ce844719e3a20562750cbfbb5088eed46ada6659ad83a",
    },
    {
        "source": "socialfx",
        "path": "socialfx/data/reverb-00000-of-00001.parquet",
        "url": "https://huggingface.co/datasets/seungheondoh/socialfx-original/resolve/d7e266cac1526ae7db52ea16c744595a05b9df27/data/reverb-00000-of-00001.parquet",
        "sha256": "36a8e56839c30fd87a15b2f653235ba19b8d156bd437bb82b02a36ae0c75c66c",
    },
    {
        "source": "timbral_hierarchy",
        "path": "timbral_hierarchy/TimbralHierarchy.zip",
        "url": "https://zenodo.org/records/167392/files/TimbralHierarchy.zip?download=1",
        "sha256": "5c0a8ad997dfa53daba92690a97ff8f10bb69c9f84354243eb3deb9adcec6327",
    },
]


if __name__ == "__main__":
    main()
