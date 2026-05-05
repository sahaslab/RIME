import os
import re
import sys
import time
import json
import shutil
import hashlib
import asyncio
import logging
import tomllib
import contextlib
from pathlib import Path
from collections.abc import Mapping, Callable, Sequence
import numpy as np
import torch
import soundfile as sf
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DARTFS_CACHE_ROOT = Path("/dartfs/rc/lab/S/SinghN/noah/.cache")

sys.path.append(str(REPO_ROOT))


def _post_master_cache_root() -> Path:
    cache_root = os.environ.get("POST_MASTER_CACHE_DIR")
    if cache_root:
        return Path(cache_root).expanduser()
    if DEFAULT_DARTFS_CACHE_ROOT.exists():
        return DEFAULT_DARTFS_CACHE_ROOT
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser() / "post-master"
    return Path.home() / ".cache" / "post-master"


def _configure_torch_home() -> None:
    torch_home = (
        os.environ.get("POST_MASTER_TORCH_HOME")
        or os.environ.get("TORCH_HOME")
        or str(_post_master_cache_root() / "torch")
    )
    torch_home_path = Path(torch_home).expanduser()
    torch_home_path.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home_path)
    torch.hub.set_dir(str(torch_home_path))


def _huggingface_cache_dir() -> Path:
    cache_dir = (
        os.environ.get("POST_MASTER_HF_CACHE_DIR")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
    )
    if cache_dir:
        return Path(cache_dir).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return _post_master_cache_root() / "huggingface" / "hub"


def _skey_checkpoint_path() -> Path:
    checkpoint_path = (
        os.environ.get("POST_MASTER_SKEY_CHECKPOINT")
        or os.environ.get("SKEY_CHECKPOINT")
    )
    if checkpoint_path:
        return Path(checkpoint_path).expanduser()
    return REPO_ROOT / "models" / "skey.pt"


_configure_torch_home()

import libraries.separation as separation_lib
from libraries.pitch import autotune, generate_harmony, apply_pitch_shift
from libraries.mixing import (
    apply_pan,
    apply_gain,
    apply_lowpass,
    apply_highpass,
    apply_lowshelf,
    normalize_peak,
    apply_highshelf,
    apply_noisegate,
    apply_peakfilter,
    apply_fade_in_out,
    mix_stem_with_residual,
)
from libraries.effects import apply_delay, apply_chorus, apply_phaser, apply_reverb, apply_deesser, apply_limiter, apply_compressor, apply_distortion
from ground_truth.runtime import RuntimePlanCompiler
from libraries.separation import separate
from audio_queue import AudioProcessingQueue
from mcp.server.fastmcp import FastMCP
from skey.key_detection import load_checkpoint, load_model_components

# Initialize FastMCP server
server = FastMCP("Audio Editing Server", "1.0.0")

# Global variables for the separation model
separation_model = None
separation_processor = None

# Global variables for pitch models
pitch_hcqt = None
pitch_chromanet = None
pitch_crop_fn = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processing_queue = AudioProcessingQueue()
log_file = "server_status.log"
DEFAULT_SERVER_CONFIG_PATH = os.environ.get(
    "POST_MASTER_SERVER_CONFIG",
    str((Path(__file__).resolve().parents[1] / "configs" / "ground_truth" / "server" / "server.toml").resolve()),
)
MIN_DEMUCS_STEM_RMS = 1e-4
MIN_DEMUCS_STEM_RMS_RATIO = 0.02
MIXTURE_HINTS = (
    "mixture",
    "mix",
    "song",
    "full track",
    "master",
    "beat",
    "backing track",
    "stems",
    "multiple sources",
)
SINGLE_SOURCE_HINTS = (
    "vocal take",
    "vocal track",
    "vocal stem",
    "isolated",
    "single source",
    "dry guitar",
    "bass recording",
    "bass track",
    "guitar recording",
    "guitar track",
    "drum stem",
    "drum track",
    "instrument track",
)
TARGET_SOURCE_HINTS = (
    "vocals",
    "vocal",
    "drums",
    "drum",
    "bass",
    "guitar",
    "piano",
    "keys",
    "synth",
    "lead",
    "backing vocals",
)
REQUESTED_EDIT_HINTS = (
    "reverb",
    "delay",
    "compress",
    "compression",
    "eq",
    "equaliz",
    "de-ess",
    "deess",
    "de-esser",
    "limit",
    "autotune",
    "harmony",
    "pitch shift",
    "distortion",
    "chorus",
    "phaser",
    "pan",
    "fade",
    "gain",
    "normalize",
    "noise gate",
)
SOURCE_CANONICAL = {
    "vocal": "vocals",
    "vocals": "vocals",
    "backing vocals": "backing vocals",
    "drum": "drums",
    "drums": "drums",
    "bass": "bass",
    "guitar": "guitar",
    "piano": "piano",
    "keys": "piano",
    "synth": "synth",
    "lead": "lead",
}
EDIT_TOOL_ALIASES = (
    ("pitch shift", "apply_pitch_shift_effect"),
    ("noise gate", "apply_noisegate_tool"),
    ("de-esser", "apply_deesser_tool"),
    ("de-ess", "apply_deesser_tool"),
    ("deess", "apply_deesser_tool"),
    ("compression", "apply_compressor_effect"),
    ("compress", "apply_compressor_effect"),
    ("reverb", "apply_reverb_effect"),
    ("chorus", "apply_chorus_effect"),
    ("delay", "apply_delay_effect"),
    ("phaser", "apply_phaser_effect"),
    ("distortion", "apply_distortion_effect"),
    ("autotune", "apply_autotune"),
    ("harmony", "apply_harmony"),
    ("equaliz", "apply_peak_filter_tool"),
    ("eq", "apply_peak_filter_tool"),
    ("limit", "apply_limiter_effect"),
    ("normalize", "normalize_peak_tool"),
    ("pan", "apply_pan_tool"),
    ("fade", "apply_fade_in_out_tool"),
    ("turn up", "apply_gain_tool"),
    ("boost", "apply_gain_tool"),
    ("gain", "apply_gain_tool"),
)
server_config: dict[str, Any] = {
    "separation_backend": "demucs",
    "demucs_model": "htdemucs_6s",
    "sam_model": "facebook/sam-audio-large",
}

# Ground-truth rendering globals
ground_truth_compiler = None
ground_truth_compiler_dir = None
ground_truth_last_audio_path = None
ground_truth_last_audio_tensor = None
ground_truth_last_sample_rate = None
ground_truth_last_max_audio_seconds = None
ground_truth_separation_cache = {}
ground_truth_demucs_sources_cache = {}
ground_truth_separation_cache_dir = None
ground_truth_raw_separate = separation_lib.separate

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _redirect_stdout_loggers_to_stderr(*logger_names: str) -> None:
    names = tuple(logger_names) or ("",)
    for logger_name in names:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers:
            stream = getattr(handler, "stream", None)
            if stream is sys.stdout:
                handler.stream = sys.stderr


_redirect_stdout_loggers_to_stderr("", "sam_audio", "mcp", "mcp.server", "mcp.server.lowlevel.server")


def load_server_config(config_path: str = DEFAULT_SERVER_CONFIG_PATH) -> dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Server config file not found: {config_path}")

    with open(config_path, "rb") as f:
        raw_config = tomllib.load(f)

    config_section = raw_config.get("server", raw_config)
    if not isinstance(config_section, dict):
        raise ValueError(
            f"Server config at {config_path} must contain a [server] table or top-level mapping"
        )

    required_fields = ["separation_backend", "demucs_model", "sam_model"]
    missing_fields = [field for field in required_fields if config_section.get(field) is None]
    if missing_fields:
        raise ValueError(
            f"Missing required server config fields in {config_path}: {', '.join(missing_fields)}"
        )

    return {
        "separation_backend": config_section["separation_backend"],
        "demucs_model": config_section["demucs_model"],
        "sam_model": config_section["sam_model"],
    }


server_config = load_server_config()


def log_message(msg: str) -> None:
    rendered = "[%s] %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(rendered, file=sys.stderr, flush=True)
    with open(log_file, "a") as f:
        f.write("%s\n" % rendered)


def _separation_runtime_ready() -> bool:
    backend = str(server_config["separation_backend"]).lower()
    if backend == "demucs":
        return separation_model is not None
    if backend == "sam_audio":
        return separation_model is not None and separation_processor is not None
    return False


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Load audio from file and return audio data and sample rate."""
    try:
        audio_data, sr = sf.read(file_path, always_2d=True, dtype="float32")
        return np.ascontiguousarray(audio_data.T), sr
    except Exception as e:
        raise ValueError(f"Error loading audio file: {str(e)}")


def load_audio_as_tensor(file_path: str) -> tuple[torch.Tensor, int]:
    """Load audio from file and return audio tensor and sample rate."""
    audio, sr = load_audio(file_path)
    return torch.from_numpy(audio.copy()), sr


def save_audio(
    audio_data: np.ndarray | torch.Tensor,
    sample_rate: int,
    output_path: str,
) -> str:
    """Save audio arrays or tensors to disk. output_path is required."""
    if not output_path or not isinstance(output_path, str):
        raise ValueError("output_path must be a non-empty string")

    if isinstance(audio_data, torch.Tensor):
        audio_array = audio_data.detach().cpu().float().numpy()
    elif isinstance(audio_data, np.ndarray):
        audio_array = np.asarray(audio_data, dtype=np.float32)
    else:
        raise ValueError("audio_data must be a numpy array or torch tensor")

    if audio_array.ndim == 1:
        audio_array = audio_array[np.newaxis, :]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, np.ascontiguousarray(audio_array.T), sample_rate)
    return output_path


def _process_simple_effect(
    *,
    audio_file: str,
    output_path: str,
    processor: Callable[..., np.ndarray | torch.Tensor],
    processor_args: tuple = (),
    processor_kwargs: dict | None = None,
) -> dict[str, Any]:
    """
    Shared handler for CPU/simple effects that operate on numpy audio.
    - loads audio (numpy)
    - calls processor(audio, sr, *args, **kwargs)
    - saves to output_path
    """
    if processor_kwargs is None:
        processor_kwargs = {}

    audio, sr = load_audio(audio_file)
    processed = processor(audio, sr, *processor_args, **processor_kwargs)
    final_path = save_audio(processed, sr, output_path)
    return {"audio_path": final_path, "sample_rate": sr}


def _guess_target_source(user_request: str) -> str | None:
    request = user_request.lower()
    for hint in TARGET_SOURCE_HINTS:
        if hint in request:
            return hint
    return None


def _guess_input_type(user_request: str, target_source: str | None) -> str:
    request = user_request.lower()
    if any(hint in request for hint in SINGLE_SOURCE_HINTS):
        return "single-source"
    if any(hint in request for hint in MIXTURE_HINTS):
        return "mixture"
    return "unknown"


def _collect_requested_edits(user_request: str) -> list[str]:
    request = user_request.lower()
    edits = [hint for hint in REQUESTED_EDIT_HINTS if hint in request]
    return edits or ["requested processing"]


def _canonicalize_target_source(text: str) -> str | None:
    text_lower = text.lower()
    for hint in sorted(TARGET_SOURCE_HINTS, key=len, reverse=True):
        if hint in text_lower:
            return SOURCE_CANONICAL.get(hint, hint)
    return None


def _extract_requested_operations(user_request: str) -> list[dict[str, str | None]]:
    request_lower = user_request.lower()
    target_mentions = [
        (match.start(), SOURCE_CANONICAL.get(hint, hint))
        for hint in sorted(TARGET_SOURCE_HINTS, key=len, reverse=True)
        for match in re.finditer(re.escape(hint), request_lower)
    ]
    target_mentions.sort(key=lambda item: item[0])

    unique_targets = []
    for _, target in target_mentions:
        if target not in unique_targets:
            unique_targets.append(target)

    operations: list[dict[str, str | None]] = []
    seen_pairs = set()
    for alias, tool_name in EDIT_TOOL_ALIASES:
        for match in re.finditer(re.escape(alias), request_lower):
            sentence_end = len(request_lower)
            for delimiter in ".;!?":
                candidate_end = request_lower.find(delimiter, match.end())
                if candidate_end != -1:
                    sentence_end = min(sentence_end, candidate_end)

            target_source = None
            for target_pos, target in target_mentions:
                if match.end() <= target_pos < sentence_end:
                    target_source = target
                    break

            if target_source is None and len(unique_targets) == 1:
                target_source = unique_targets[0]

            pair_key = (match.start(), tool_name, target_source)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            operations.append(
                {
                    "position": match.start(),
                    "alias": alias,
                    "tool": tool_name,
                    "target": target_source,
                }
            )

    operations.sort(key=lambda item: int(item["position"]))
    return operations


def _group_operations_by_target(
    operations: list[dict[str, str | None]],
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for operation in operations:
        target = operation.get("target")
        if groups and groups[-1]["target"] == target:
            groups[-1]["operations"].append(operation)
            continue
        groups.append({"target": target, "operations": [operation]})
    return groups


def _build_plan_text(user_request: str) -> str:
    target_source = _guess_target_source(user_request)
    input_type = _guess_input_type(user_request, target_source)
    requested_edits = ", ".join(_collect_requested_edits(user_request))
    operations = _extract_requested_operations(user_request)
    operation_groups = _group_operations_by_target(operations)

    steps = [
        "1. Listen to the current audio and confirm whether it is truly a full mix or already an isolated source before committing to separation.",
    ]

    if input_type == "mixture" and operation_groups:
        step_number = 2
        for group in operation_groups:
            group_target = group["target"]
            group_operations = group["operations"]

            if group_target:
                steps.append(
                    f"{step_number}. Likely tool: separate_audio(audio_file=CURRENT_AUDIO, description=\"{group_target}\", return_both=true) to isolate the target before processing."
                )
                step_number += 1
                for operation in group_operations:
                    steps.append(
                        f"{step_number}. Likely tool: {operation['tool']}(audio_file=STEM_PATH, output_path=REQUEST_DIR + \"/...\") for the {group_target} stem."
                    )
                    step_number += 1
                steps.append(
                    f"{step_number}. Likely tool: mix_sources(processed_stem_file=PROCESSED_STEM_PATH, residual_file=RESIDUAL_PATH, output_path=REQUEST_DIR + \"/...\") to rebuild the full mix before the next edit."
                )
                step_number += 1
            else:
                for operation in group_operations:
                    steps.append(
                        f"{step_number}. Likely tool: {operation['tool']}(audio_file=CURRENT_AUDIO, output_path=REQUEST_DIR + \"/...\") directly on the current mix."
                    )
                    step_number += 1

        steps.append(
            f"{step_number}. Likely tool: return_audio(audio_file=CURRENT_AUDIO, message=\"...\") once the remixed result satisfies the request."
        )
    elif input_type == "unknown":
        steps.extend(
            [
                "2. Do not assume this is a mixture just because the request names a source like vocals or bass.",
                "3. If listening confirms a full mix, use separate_audio for each targeted source, apply the matching effect tools to STEM_PATH, then use mix_sources after each source-specific edit.",
                "4. If listening confirms an isolated source, apply the matching effect tools directly to CURRENT_AUDIO in sequence without separation.",
                "5. After each edit, re-listen and deviate from this plan whenever the latest audio suggests a better sequence or parameter choice.",
                "6. Finish with return_audio once the strongest result matches the request.",
            ]
        )
    else:
        step_number = 2
        if operation_groups:
            for group in operation_groups:
                for operation in group["operations"]:
                    steps.append(
                        f"{step_number}. Likely tool: {operation['tool']}(audio_file=CURRENT_AUDIO, output_path=REQUEST_DIR + \"/...\") directly on the isolated source."
                    )
                    step_number += 1
        else:
            steps.append(
                f"{step_number}. Process the current audio directly, applying {requested_edits} one step at a time."
            )
            step_number += 1
        steps.append(
            f"{step_number}. Likely tool: return_audio(audio_file=CURRENT_AUDIO, message=\"...\") once the result satisfies the request."
        )

    return "\n".join(steps)


@server.tool()
def plan_edit_sequence(
    user_request: str,
    audio_file: str | None = None,
) -> dict:
    """Create an advisory natural-language edit plan before running audio edits."""
    return {
        "plan": _build_plan_text(user_request),
        "user_request": user_request,
        "audio_file": audio_file,
        "advisory_only": True,
    }


# ============================================================================
# SEPARATION TOOLS
# ============================================================================


async def initialize_separation_models():
    if not processing_queue.is_running:
        await processing_queue.start()

    global separation_model, separation_processor
    separation_backend = str(server_config["separation_backend"]).lower()

    if separation_backend == "demucs":
        if separation_model is not None:
            log_message("Separation model already initialized with Demucs.")
            return
        try:
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise ImportError(
                "Demucs is not installed. Install the `demucs` package to use separation."
            ) from exc

        model_name = str(server_config["demucs_model"])
        log_message("Loading Demucs separation model (%s) on %s." % (model_name, device))
        separation_model = get_model(name=model_name).to(device).eval()
        separation_processor = None
        log_message("Separation model initialized with Demucs (%s)." % model_name)
        return

    if separation_backend == "sam_audio":
        if separation_model is not None and separation_processor is not None:
            log_message("Separation model already initialized with SAM Audio.")
            return
        try:
            from sam_audio import SAMAudio, SAMAudioProcessor
            from sam_audio.model.config import SAMAudioConfig
        except ImportError as exc:
            raise ImportError(
                "SAM Audio is not installed. Install the `sam_audio` package to use separation."
            ) from exc

        cache_dir = _huggingface_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cfg = SAMAudioConfig(visual_ranker=None)
        model_name = str(server_config["sam_model"])
        log_message("Loading SAM Audio separation model (%s) on %s." % (model_name, device))
        if device.type == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32

        with contextlib.redirect_stdout(sys.stderr):
            separation_model = (
                SAMAudio.from_pretrained(
                    model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    cache_dir=str(cache_dir),
                    config=cfg,
                )
                .to(device)
                .eval()
            )
            separation_processor = SAMAudioProcessor.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
            )
        log_message("Separation model initialized with SAM Audio (%s)." % model_name)
        return

    raise ValueError(
        f"Unsupported separation_backend '{server_config['separation_backend']}'. "
        "Expected 'demucs' or 'sam_audio'."
    )


def _separate_audio_logic(
    audio_file: str,
    description: str,
    return_both: bool,
    output_path_stem: str,
    output_path_residual: str,
):
    if separation_model is None:
        raise ValueError("Separation model not initialized.")

    audio_tensor, sr = load_audio_as_tensor(audio_file)
    stem, residual = separate(
        separation_model,
        separation_processor,
        device,
        audio_tensor,
        description,
        sample_rate=sr,
    )

    saved_stem_path = save_audio(stem, sr, output_path_stem)

    if return_both:
        saved_residual_path = save_audio(residual, sr, output_path_residual)
        return {
            "stem_path": saved_stem_path,
            "residual_path": saved_residual_path,
            "sample_rate": sr,
        }

    return {"stem_path": saved_stem_path, "sample_rate": sr}


@server.tool()
async def separate_audio(
    audio_file: str,
    description: str,
    output_path_stem: str,
    output_path_residual: str,
    return_both: bool = True,
) -> dict:
    """Separate audio into stem and residual based on description."""
    if separation_model is None:
        await initialize_separation_models()

    return await processing_queue.enqueue(
        _separate_audio_logic,
        audio_file=audio_file,
        description=description,
        return_both=return_both,
        output_path_stem=output_path_stem,
        output_path_residual=output_path_residual,
    )


# ============================================================================
# EFFECTS TOOLS
# ============================================================================


@server.tool()
def apply_chorus_effect(
    audio_file: str,
    output_path: str,
    rate_hz: float = 1.0,
    depth: float = 0.25,
    centre_delay_ms: float = 7.0,
    feedback: float = 0.0,
    mix: float = 0.5,
) -> dict:
    """Apply chorus effect to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_chorus,
        processor_args=(rate_hz, depth, centre_delay_ms, feedback, mix),
    )


@server.tool()
def apply_phaser_effect(
    audio_file: str,
    output_path: str,
    rate_hz: float = 1.0,
    depth: float = 0.5,
    centre_frequency_hz: float = 1300.0,
    feedback: float = 0.0,
    mix: float = 0.5,
) -> dict:
    """Apply phaser effect to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_phaser,
        processor_args=(rate_hz, depth, centre_frequency_hz, feedback, mix),
    )


@server.tool()
def apply_distortion_effect(
    audio_file: str,
    output_path: str,
    drive_db: float = 25.0,
) -> dict:
    """Apply distortion effect to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_distortion,
        processor_args=(drive_db,),
    )


@server.tool()
def apply_pitch_shift_effect(
    audio_file: str,
    output_path: str,
    semitones: float = 0.0,
) -> dict:
    """Apply a constant pitch shift to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_pitch_shift,
        processor_args=(semitones,),
    )


@server.tool()
def apply_reverb_effect(
    audio_file: str,
    output_path: str,
    room_size: float = 0.5,
    damping: float = 0.5,
    wet_level: float = 0.33,
    dry_level: float = 0.4,
    width: float = 1.0,
    freeze_mode: float = 0.0,
) -> dict:
    """Apply reverb effect to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_reverb,
        processor_args=(room_size, damping, wet_level, dry_level, width, freeze_mode),
    )


@server.tool()
def apply_delay_effect(
    audio_file: str,
    output_path: str,
    delay_seconds: float = 0.5,
    feedback: float = 0.0,
    mix: float = 0.5,
) -> dict:
    """Apply delay effect to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_delay,
        processor_args=(delay_seconds, feedback, mix),
    )


@server.tool()
def apply_compressor_effect(
    audio_file: str,
    output_path: str,
    threshold_db: float = 0.0,
    ratio: float = 1.0,
    attack_ms: float = 1.0,
    release_ms: float = 10.0,
) -> dict:
    """Apply compressor to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_compressor,
        processor_args=(threshold_db, ratio, attack_ms, release_ms),
    )


@server.tool()
def apply_limiter_effect(
    audio_file: str,
    output_path: str,
    threshold_db: float = -10.0,
    release_ms: float = 100.0,
) -> dict:
    """Apply limiter to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_limiter,
        processor_args=(threshold_db, release_ms),
    )


@server.tool()
def apply_deesser_tool(
    audio_file: str,
    output_path: str,
    ess_highpass_hz: float = 5000.0,
    ess_lowpass_hz: float = 10000.0,
    threshold_db: float = -30.0,
    ratio: float = 8.0,
    attack_ms: float = 1.0,
    release_ms: float = 60.0,
) -> dict:
    """Apply a simple split-band de-esser to reduce sibilance."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_deesser,
        processor_args=(
            ess_highpass_hz,
            ess_lowpass_hz,
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
        ),
    )


# ============================================================================
# MIXING TOOLS
# ============================================================================


@server.tool()
def apply_gain_tool(
    audio_file: str,
    output_path: str,
    gain_db: float = 1.0,
) -> dict:
    """Apply gain to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_gain,
        processor_args=(gain_db,),
    )


@server.tool()
def apply_highpass_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: float = 50.0,
) -> dict:
    """Apply highpass filter to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_highpass,
        processor_args=(cutoff_frequency_hz,),
    )


@server.tool()
def apply_lowpass_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: float = 50.0,
) -> dict:
    """Apply lowpass filter to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_lowpass,
        processor_args=(cutoff_frequency_hz,),
    )


@server.tool()
def apply_highshelf_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: float = 440.0,
    gain_db: float = 0.0,
    q: float = 0.7071067690849304,
) -> dict:
    """Apply high shelf filter to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_highshelf,
        processor_args=(cutoff_frequency_hz, gain_db, q),
    )


@server.tool()
def apply_lowshelf_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: float = 440.0,
    gain_db: float = 0.0,
    q: float = 0.7071067690849304,
) -> dict:
    """Apply low shelf filter to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_lowshelf,
        processor_args=(cutoff_frequency_hz, gain_db, q),
    )


@server.tool()
def apply_peak_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: float = 440.0,
    gain_db: float = 0.0,
    q: float = 0.7071067690849304,
) -> dict:
    """Apply peak filter (parametric EQ) to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_peakfilter,
        processor_args=(cutoff_frequency_hz, gain_db, q),
    )


@server.tool()
def apply_noisegate_tool(
    audio_file: str,
    output_path: str,
    threshold_db: float = -100.0,
    ratio: float = 10.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
) -> dict:
    """Apply noise gate to audio."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_noisegate,
        processor_args=(threshold_db, ratio, attack_ms, release_ms),
    )


@server.tool()
def normalize_peak_tool(
    audio_file: str,
    output_path: str,
    target_peak: float = 0.95,
) -> dict:
    """Scale audio so the peak sample reaches the requested headroom target."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=normalize_peak,
        processor_args=(target_peak,),
    )


@server.tool()
def apply_fade_in_out_tool(
    audio_file: str,
    output_path: str,
    fade_in_seconds: float = 0.01,
    fade_out_seconds: float = 0.01,
) -> dict:
    """Apply linear fade-in and fade-out to the provided audio file."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_fade_in_out,
        processor_args=(fade_in_seconds, fade_out_seconds),
    )


@server.tool()
def apply_pan_tool(
    audio_file: str,
    output_path: str,
    pan: float = 0.0,
) -> dict:
    """Apply constant-power stereo panning. Negative is left, positive is right."""
    return _process_simple_effect(
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_pan,
        processor_args=(pan,),
    )


@server.tool()
def mix_sources(
    processed_stem_file: str,
    residual_file: str,
    output_path: str,
) -> dict:
    """Recombine processed stem + residual into a final mix."""
    stem, sr1 = load_audio(processed_stem_file)
    residual, sr2 = load_audio(residual_file)

    if sr1 != sr2:
        raise ValueError("Sample rates must match")
    if stem.shape != residual.shape:
        raise ValueError("Audio shapes must match")

    mixed = mix_stem_with_residual(stem, residual)
    final_path = save_audio(mixed, sr1, output_path)

    return {"audio_path": final_path, "sample_rate": sr1}


# ============================================================================
# PITCH TOOLS
# ============================================================================


async def initialize_pitch_models():
    if not processing_queue.is_running:
        await processing_queue.start()
    global pitch_hcqt, pitch_chromanet, pitch_crop_fn
    if not any(m is None for m in [pitch_hcqt, pitch_chromanet, pitch_crop_fn]):
        log_message("Pitch models already initialized.")
        return
    checkpoint_path = _skey_checkpoint_path()
    log_message("Loading pitch models from %s on %s." % (checkpoint_path, device))
    with contextlib.redirect_stdout(sys.stderr):
        ckpt = load_checkpoint(str(checkpoint_path))
        pitch_hcqt, pitch_chromanet, pitch_crop_fn = load_model_components(ckpt, device)
    log_message("Pitch models initialized.")


def _autotune_logic(
    audio_file: str,
    key: str | None,
    mode: str | None,
    output_path: str,
):
    audio_tensor, sr = load_audio_as_tensor(audio_file)
    processed = autotune(
        audio=audio_tensor.to(device),
        sr=sr,
        hcqt=pitch_hcqt,
        chromanet=pitch_chromanet,
        crop_fn=pitch_crop_fn,
        device=device,
        key=key,
        mode=mode,
    )

    final_path = save_audio(processed, sr, output_path)
    return {"audio_path": final_path, "sample_rate": sr}


def _harmony_logic(
    audio_file: str,
    semitones: float,
    key: str | None,
    mode: str | None,
    output_path: str,
):
    audio_tensor, sr = load_audio_as_tensor(audio_file)
    processed = generate_harmony(
        audio=audio_tensor.to(device),
        sr=sr,
        hcqt=pitch_hcqt,
        chromanet=pitch_chromanet,
        crop_fn=pitch_crop_fn,
        device=device,
        semitones=semitones,
        key=key,
        mode=mode,
    )

    final_path = save_audio(processed, sr, output_path)
    return {"audio_path": final_path, "sample_rate": sr}


@server.tool()
async def apply_autotune(
    audio_file: str,
    output_path: str,
    key: str | None = None,
    mode: str | None = None,
) -> dict:
    """Quantize input vocals to a key. If key/mode omitted, they will be inferred."""
    if any(m is None for m in [pitch_hcqt, pitch_chromanet, pitch_crop_fn]):
        await initialize_pitch_models()

    return await processing_queue.enqueue(
        _autotune_logic,
        audio_file=audio_file,
        key=key,
        mode=mode,
        output_path=output_path,
    )


@server.tool()
async def apply_harmony(
    audio_file: str,
    semitones: float,
    output_path: str,
    key: str | None = None,
    mode: str | None = None,
) -> dict:
    """Generate a harmony at a specified semitone distance; returns harmony-only."""
    if any(m is None for m in [pitch_hcqt, pitch_chromanet, pitch_crop_fn]):
        await initialize_pitch_models()

    return await processing_queue.enqueue(
        _harmony_logic,
        audio_file=audio_file,
        semitones=semitones,
        key=key,
        mode=mode,
        output_path=output_path,
    )


# ============================================================================
# GROUND-TRUTH RENDERING
# ============================================================================


def _ground_truth_config_dir(config_dir: str | None) -> Path:
    if config_dir:
        return Path(config_dir).expanduser().resolve()
    return (Path(__file__).resolve().parents[1] / "configs" / "ground_truth").resolve()


def _ground_truth_compiler_for(config_dir: str | None) -> RuntimePlanCompiler:
    global ground_truth_compiler, ground_truth_compiler_dir
    resolved = str(_ground_truth_config_dir(config_dir))
    if ground_truth_compiler is None or ground_truth_compiler_dir != resolved:
        ground_truth_compiler = RuntimePlanCompiler.from_directory(resolved)
        ground_truth_compiler_dir = resolved
    return ground_truth_compiler


def _ensure_ground_truth_audio(
    audio_file: str,
    max_audio_seconds: float | None = None,
) -> tuple[torch.Tensor, int]:
    global ground_truth_last_audio_path
    global ground_truth_last_audio_tensor
    global ground_truth_last_sample_rate
    global ground_truth_last_max_audio_seconds
    global ground_truth_separation_cache
    global ground_truth_demucs_sources_cache

    resolved = str(Path(audio_file).expanduser().resolve())
    normalized_max_audio_seconds = (
        None
        if max_audio_seconds is None or float(max_audio_seconds) <= 0.0
        else float(max_audio_seconds)
    )
    if (
        ground_truth_last_audio_path == resolved
        and ground_truth_last_max_audio_seconds == normalized_max_audio_seconds
        and ground_truth_last_audio_tensor is not None
        and ground_truth_last_sample_rate is not None
    ):
        return ground_truth_last_audio_tensor, ground_truth_last_sample_rate

    audio_tensor, sample_rate = load_audio_as_tensor(resolved)
    if normalized_max_audio_seconds is not None:
        max_samples = int(normalized_max_audio_seconds * sample_rate)
        if max_samples > 0 and audio_tensor.shape[-1] > max_samples:
            audio_tensor = audio_tensor[..., :max_samples].contiguous()
    ground_truth_last_audio_path = resolved
    ground_truth_last_audio_tensor = audio_tensor
    ground_truth_last_sample_rate = sample_rate
    ground_truth_last_max_audio_seconds = normalized_max_audio_seconds
    ground_truth_separation_cache = {}
    ground_truth_demucs_sources_cache = {}
    return audio_tensor, sample_rate


def _set_ground_truth_separation_cache_dir(cache_dir: str | None) -> None:
    global ground_truth_separation_cache_dir
    if cache_dir is None or str(cache_dir).strip() == "":
        ground_truth_separation_cache_dir = None
        return
    resolved = Path(cache_dir).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    ground_truth_separation_cache_dir = resolved


def _ground_truth_cache_owner(audio: torch.Tensor) -> str:
    if ground_truth_last_audio_tensor is audio and ground_truth_last_audio_path is not None:
        return str(ground_truth_last_audio_path)
    return "tensor:%d" % id(audio)


def _ground_truth_disk_cache_path(
    *,
    prefix: str,
    audio: torch.Tensor,
    description: str | None,
    sr: int | None,
) -> Path | None:
    if ground_truth_separation_cache_dir is None:
        return None
    if ground_truth_last_audio_tensor is not audio or ground_truth_last_audio_path is None:
        return None

    audio_path = Path(ground_truth_last_audio_path)
    audio_stat = audio_path.stat()
    payload = {
        "audio_path": str(audio_path),
        "audio_size": audio_stat.st_size,
        "audio_mtime_ns": audio_stat.st_mtime_ns,
        "description": description,
        "sample_rate": sr,
        "max_audio_seconds": ground_truth_last_max_audio_seconds,
        "separation_backend": server_config.get("separation_backend"),
        "demucs_model": server_config.get("demucs_model"),
        "demucs_overlap": float(os.environ.get("DEMUCS_OVERLAP", "0.5")),
        "sam_model": server_config.get("sam_model"),
        "prefix": prefix,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return ground_truth_separation_cache_dir / prefix / ("%s.pt" % digest)


def _clone_source_map(source_map: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        str(source_name): source_audio.detach().cpu().clone()
        for source_name, source_audio in source_map.items()
    }


def _sum_sources_except(
    source_map: Mapping[str, torch.Tensor],
    target_source: str,
) -> torch.Tensor:
    residual_sources = [
        source_audio
        for source_name, source_audio in source_map.items()
        if source_name != target_source
    ]
    return torch.stack(residual_sources, dim=0).sum(dim=0)


def _sum_sources(source_map: Mapping[str, torch.Tensor]) -> torch.Tensor:
    source_iter = iter(source_map.values())
    mixture = next(source_iter).detach().cpu().float().clone()
    for source_audio in source_iter:
        mixture.add_(source_audio.detach().cpu().float())
    return mixture


def _audio_rms(audio: torch.Tensor) -> float:
    audio_cpu = audio.detach().cpu().float()
    return float(torch.sqrt(torch.mean(audio_cpu * audio_cpu)).item())


def _assert_active_demucs_source(
    source_map: Mapping[str, torch.Tensor],
    target_source: str,
) -> None:
    started_at = time.perf_counter()
    log_message("Demucs activity check start | target_source=%s" % target_source)
    source_rms = _audio_rms(source_map[target_source])
    log_message("Demucs activity check source RMS | target_source=%s | rms=%.6f" % (target_source, source_rms))
    mixture_rms = _audio_rms(_sum_sources(source_map))
    rms_ratio = source_rms / max(mixture_rms, MIN_DEMUCS_STEM_RMS)
    log_message(
        "Demucs activity check done | target_source=%s | rms=%.6f | mixture_rms=%.6f | rms_ratio=%.6f | elapsed=%.1fs"
        % (
            target_source,
            source_rms,
            mixture_rms,
            rms_ratio,
            time.perf_counter() - started_at,
        )
    )
    if source_rms < MIN_DEMUCS_STEM_RMS or rms_ratio < MIN_DEMUCS_STEM_RMS_RATIO:
        raise ValueError(
            "Demucs target '%s' is below activity threshold: rms=%.6f rms_ratio=%.6f"
            % (target_source, source_rms, rms_ratio)
        )


def _ground_truth_cached_demucs_sources(
    model: Any,
    device: torch.device,
    audio: torch.Tensor,
    sr: int | None,
) -> dict[str, torch.Tensor]:
    cache_owner = _ground_truth_cache_owner(audio)
    cache_key = (str(cache_owner), "demucs_sources", str(sr), str(server_config.get("demucs_model")))
    if cache_key not in ground_truth_demucs_sources_cache:
        disk_cache_path = _ground_truth_disk_cache_path(
            prefix="demucs_sources",
            audio=audio,
            description=None,
            sr=sr,
        )
        if disk_cache_path is not None and disk_cache_path.exists():
            log_message("Demucs source cache hit | path=%s" % disk_cache_path)
            payload = torch.load(disk_cache_path, map_location="cpu")
            source_map = _clone_source_map(payload["sources"])
        else:
            log_message(
                "Demucs source cache miss | owner=%s | sr=%s | samples=%d | cache_path=%s"
                % (
                    cache_owner,
                    sr,
                    audio.shape[-1],
                    disk_cache_path,
                )
            )
            source_map = separation_lib.separate_demucs_sources(model, device, audio, sr)
            source_map = _clone_source_map(source_map)
            if disk_cache_path is not None:
                disk_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"sources": source_map}, disk_cache_path)
                log_message("Demucs source cache written | path=%s" % disk_cache_path)
        ground_truth_demucs_sources_cache[cache_key] = source_map
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        log_message("Demucs source memory cache hit | owner=%s | sr=%s" % (cache_owner, sr))
    return _clone_source_map(ground_truth_demucs_sources_cache[cache_key])


def _ground_truth_cached_separate(
    model: Any,
    processor: Any,
    device: torch.device,
    audio: torch.Tensor,
    description: str,
    sr: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(model, "sources"):
        source_map = _ground_truth_cached_demucs_sources(model, device, audio, sr)
        target_source = separation_lib.canonicalize_demucs_description(
            description,
            list(source_map.keys()),
        )
        log_message("Selecting Demucs target | description=%s | target_source=%s" % (description, target_source))
        _assert_active_demucs_source(source_map, target_source)
        stem = source_map[target_source]
        residual = _sum_sources_except(source_map, target_source)
        return stem.clone(), residual.clone()

    cache_owner = _ground_truth_cache_owner(audio)
    cache_key = (str(cache_owner), str(description), str(sr))
    if cache_key not in ground_truth_separation_cache:
        disk_cache_path = _ground_truth_disk_cache_path(
            prefix="separations",
            audio=audio,
            description=description,
            sr=sr,
        )
        if disk_cache_path is not None and disk_cache_path.exists():
            log_message("Separation cache hit | description=%s | path=%s" % (description, disk_cache_path))
            payload = torch.load(disk_cache_path, map_location="cpu")
            stem = payload["stem"]
            residual = payload["residual"]
        else:
            log_message(
                "Separation cache miss | description=%s | sr=%s | samples=%d | cache_path=%s"
                % (description, sr, audio.shape[-1], disk_cache_path)
            )
            stem, residual = ground_truth_raw_separate(model, processor, device, audio, description, sr=sr)
            if disk_cache_path is not None:
                disk_cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "stem": stem.detach().cpu().clone(),
                        "residual": residual.detach().cpu().clone(),
                    },
                    disk_cache_path,
                )
                log_message("Separation cache written | description=%s | path=%s" % (description, disk_cache_path))
        ground_truth_separation_cache[cache_key] = (
            stem.detach().cpu().clone(),
            residual.detach().cpu().clone(),
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        log_message("Separation memory cache hit | description=%s | sr=%s" % (description, sr))
    stem, residual = ground_truth_separation_cache[cache_key]
    return stem.clone(), residual.clone()


def _patch_ground_truth_runtime() -> None:
    separation_lib.separate = _ground_truth_cached_separate


def _graph_uses_separation(graph_spec: Sequence[Mapping[str, Any]]) -> bool:
    return any(block.get("kind") == "separate" for block in graph_spec)


def _graph_uses_pitch(graph_spec: Sequence[Mapping[str, Any]]) -> bool:
    pitch_ops = {"apply_autotune", "apply_harmony_effect"}
    for block in graph_spec:
        if block.get("kind") == "step" and block.get("operator") in pitch_ops:
            return True
        for step in block.get("steps", []):
            if step.get("operator") in pitch_ops:
                return True
    return False


def _build_runtime_context(
    graph_spec: Sequence[Mapping[str, Any]],
    audio_tensor: torch.Tensor,
    sample_rate: int,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "audio": audio_tensor,
        "sr": sample_rate,
        "device": device,
    }
    if _graph_uses_separation(graph_spec):
        if not _separation_runtime_ready():
            raise ValueError("Separation model not initialized.")
        context["model"] = separation_model
        context["processor"] = separation_processor
    if _graph_uses_pitch(graph_spec):
        if any(m is None for m in [pitch_hcqt, pitch_chromanet, pitch_crop_fn]):
            raise ValueError("Pitch model not initialized.")
        context["hcqt"] = pitch_hcqt
        context["chromanet"] = pitch_chromanet
        context["crop_fn"] = pitch_crop_fn
    return context


def _final_output_key(graph_spec: Sequence[Mapping[str, Any]]) -> str:
    if not graph_spec:
        raise ValueError("graph_spec is empty.")
    block = graph_spec[-1]
    outputs = block.get("output", block.get("outputs"))
    if outputs is None:
        return str(block["name"])
    if isinstance(outputs, str):
        return outputs
    if isinstance(outputs, Sequence) and outputs:
        return str(outputs[0])
    raise ValueError("Could not determine final output key from graph_spec.")


def _build_baseline_graph_spec(graph_spec: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    aliases: dict[str, str] = {}
    baseline_spec: list[dict[str, Any]] = []

    def resolve(name: Any) -> Any:
        if not isinstance(name, str):
            return name
        while name in aliases:
            name = aliases[name]
        return name

    for block in graph_spec:
        kind = block.get("kind")
        if kind == "separate":
            baseline_spec.append(dict(block))
            continue
        if kind in {"chain", "send_return"}:
            aliases[str(block.get("output"))] = str(resolve(block.get("source")))
            continue
        if kind == "step":
            inputs = {key: resolve(value) for key, value in dict(block.get("inputs", {})).items()}
            passthrough_source = next(iter(inputs.values()), None)
            outputs = block.get("outputs")
            if isinstance(outputs, str) and passthrough_source is not None:
                aliases[outputs] = str(passthrough_source)
                continue
            if isinstance(outputs, Sequence) and not isinstance(outputs, str) and passthrough_source is not None:
                for output_name in outputs:
                    aliases[str(output_name)] = str(passthrough_source)
                continue
            raise ValueError(f"Cannot build no-op baseline for step block '{block.get('name')}'.")
        if kind == "mix":
            copied = dict(block)
            copied["stem"] = resolve(block.get("stem"))
            copied["residual"] = resolve(block.get("residual"))
            baseline_spec.append(copied)
            continue
        raise ValueError(f"Unsupported graph block kind '{kind}' while building baseline.")
    return baseline_spec


def _execute_graph_spec(
    *,
    compiler: RuntimePlanCompiler,
    graph_spec: Sequence[Mapping[str, Any]],
    audio_tensor: torch.Tensor,
    sample_rate: int,
    output_path: Path | None = None,
) -> Any:
    output_label = "<memory>" if output_path is None else str(output_path)
    log_message(
        "Render graph start | output=%s | blocks=%d | separation=%s | pitch=%s | samples=%d | sr=%d"
        % (
            output_label,
            len(graph_spec),
            _graph_uses_separation(graph_spec),
            _graph_uses_pitch(graph_spec),
            audio_tensor.shape[-1],
            sample_rate,
        )
    )
    log_message("Render graph compile | output=%s" % output_label)
    graph = compiler.compile_graph_spec(graph_spec)
    log_message("Render graph context | output=%s" % output_label)
    context = _build_runtime_context(graph_spec, audio_tensor, sample_rate)
    log_message("Render graph execute | output=%s" % output_label)
    outputs = graph.run(**context)
    final_key = _final_output_key(graph_spec)
    if final_key not in outputs:
        raise KeyError(f"Expected final output '{final_key}' was not produced.")
    return outputs[final_key]


def _render_graph_spec(
    *,
    compiler: RuntimePlanCompiler,
    graph_spec: Sequence[Mapping[str, Any]],
    audio_tensor: torch.Tensor,
    sample_rate: int,
    output_path: Path,
) -> None:
    started_at = time.perf_counter()
    log_message(
        "Render graph start | output=%s | blocks=%d | separation=%s | pitch=%s | samples=%d | sr=%d"
        % (
            output_path,
            len(graph_spec),
            _graph_uses_separation(graph_spec),
            _graph_uses_pitch(graph_spec),
            audio_tensor.shape[-1],
            sample_rate,
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_audio = _execute_graph_spec(
        compiler=compiler,
        graph_spec=graph_spec,
        audio_tensor=audio_tensor,
        sample_rate=sample_rate,
        output_path=output_path,
    )
    final_key = _final_output_key(graph_spec)
    log_message("Render graph save | output=%s | final_key=%s" % (output_path, final_key))
    save_audio(rendered_audio, sample_rate, str(output_path))
    log_message(
        "Render graph done | output=%s | elapsed=%.1fs"
        % (output_path, time.perf_counter() - started_at)
    )


def _render_ground_truth_plan_logic(
    *,
    audio_file: str,
    graph_spec: Sequence[Mapping[str, Any]],
    output_path: str,
    poison_graph_spec: Sequence[Mapping[str, Any]] | None = None,
    config_dir: str | None = None,
    separation_cache_dir: str | None = None,
    max_audio_seconds: float | None = None,
    overwrite: bool = False,
    baseline_output_path: str | None = None,
    source_copy_path: str | None = None,
) -> dict[str, Any]:
    plan_started_at = time.perf_counter()
    source_path = Path(audio_file).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Input audio file does not exist: {source_path}")
    normalized_max_audio_seconds = (
        None
        if max_audio_seconds is None or float(max_audio_seconds) <= 0.0
        else float(max_audio_seconds)
    )

    output_path_obj = Path(output_path).expanduser().resolve()
    baseline_path_obj = Path(baseline_output_path).expanduser().resolve() if baseline_output_path else None
    source_copy_obj = Path(source_copy_path).expanduser().resolve() if source_copy_path else None

    if source_copy_obj is not None:
        source_copy_obj.parent.mkdir(parents=True, exist_ok=True)
        if normalized_max_audio_seconds is None and (overwrite or not source_copy_obj.exists()):
            shutil.copy2(source_path, source_copy_obj)

    if (
        output_path_obj.exists()
        and not overwrite
        and (baseline_path_obj is None or baseline_path_obj.exists())
        and (source_copy_obj is None or source_copy_obj.exists())
    ):
        log_message("Render skipped existing | output=%s" % output_path_obj)
        return {
            "audio_path": str(source_path),
            "output_path": str(output_path_obj),
            "baseline_path": str(baseline_path_obj) if baseline_path_obj else None,
            "source_copy_path": str(source_copy_obj) if source_copy_obj else None,
            "status": "skipped_existing",
        }

    log_message(
        "Render plan start | audio=%s | output=%s | baseline=%s | max_audio_seconds=%s | overwrite=%s"
        % (
            source_path,
            output_path_obj,
            baseline_path_obj,
            normalized_max_audio_seconds,
            overwrite,
        )
    )
    compiler = _ground_truth_compiler_for(config_dir)
    _set_ground_truth_separation_cache_dir(separation_cache_dir)
    _patch_ground_truth_runtime()
    log_message("Loading render audio | audio=%s" % source_path)
    audio_tensor, sample_rate = _ensure_ground_truth_audio(
        str(source_path),
        max_audio_seconds=normalized_max_audio_seconds,
    )
    log_message(
        "Loaded render audio | audio=%s | samples=%d | sr=%d"
        % (source_path, audio_tensor.shape[-1], sample_rate)
    )
    graph_spec_list = [dict(block) for block in graph_spec]
    poison_graph_spec_list = (
        None if poison_graph_spec is None else [dict(block) for block in poison_graph_spec]
    )

    if source_copy_obj is not None and (overwrite or not source_copy_obj.exists()):
        log_message("Writing source copy | output=%s" % source_copy_obj)
        save_audio(audio_tensor, sample_rate, str(source_copy_obj))

    render_input_audio = audio_tensor
    if poison_graph_spec_list:
        log_message(
            "Rendering poisoned initial | output=%s"
            % (baseline_path_obj if baseline_path_obj is not None else "<memory>")
        )
        render_input_audio = _execute_graph_spec(
            compiler=compiler,
            graph_spec=poison_graph_spec_list,
            audio_tensor=audio_tensor,
            sample_rate=sample_rate,
            output_path=baseline_path_obj,
        )
        if baseline_path_obj is not None and (overwrite or not baseline_path_obj.exists()):
            baseline_path_obj.parent.mkdir(parents=True, exist_ok=True)
            save_audio(render_input_audio, sample_rate, str(baseline_path_obj))
    elif baseline_path_obj is not None and (overwrite or not baseline_path_obj.exists()):
        baseline_spec = _build_baseline_graph_spec(graph_spec_list)
        if baseline_spec:
            log_message("Rendering baseline | output=%s" % baseline_path_obj)
            _render_graph_spec(
                compiler=compiler,
                graph_spec=baseline_spec,
                audio_tensor=audio_tensor,
                sample_rate=sample_rate,
                output_path=baseline_path_obj,
            )
        else:
            log_message("Writing baseline source passthrough | output=%s" % baseline_path_obj)
            save_audio(audio_tensor, sample_rate, str(baseline_path_obj))

    if overwrite or not output_path_obj.exists():
        log_message("Rendering final plan | output=%s" % output_path_obj)
        _render_graph_spec(
            compiler=compiler,
            graph_spec=graph_spec_list,
            audio_tensor=render_input_audio,
            sample_rate=sample_rate,
            output_path=output_path_obj,
        )

    log_message(
        "Render plan done | output=%s | elapsed=%.1fs"
        % (output_path_obj, time.perf_counter() - plan_started_at)
    )
    return {
        "audio_path": str(source_path),
        "output_path": str(output_path_obj),
        "baseline_path": str(baseline_path_obj) if baseline_path_obj else None,
        "source_copy_path": str(source_copy_obj) if source_copy_obj else None,
        "separation_cache_dir": str(ground_truth_separation_cache_dir) if ground_truth_separation_cache_dir else None,
        "max_audio_seconds": normalized_max_audio_seconds,
        "sample_rate": sample_rate,
        "status": "rendered",
    }


@server.tool()
async def render_ground_truth_plan(
    audio_file: str,
    graph_spec: list[dict],
    output_path: str,
    poison_graph_spec: list[dict] | None = None,
    config_dir: str | None = None,
    separation_cache_dir: str | None = None,
    max_audio_seconds: float | None = None,
    overwrite: bool = False,
    baseline_output_path: str | None = None,
    source_copy_path: str | None = None,
) -> dict:
    """Render a ground-truth graph_spec through the backend queue using long-lived server resources."""
    if not processing_queue.is_running:
        await processing_queue.start()

    required_graph_specs = [graph_spec]
    if poison_graph_spec:
        required_graph_specs.append(poison_graph_spec)

    if any(_graph_uses_separation(spec) for spec in required_graph_specs) and not _separation_runtime_ready():
        log_message("Render request requires separation; initializing separation models.")
        await initialize_separation_models()
    if any(_graph_uses_pitch(spec) for spec in required_graph_specs) and any(m is None for m in [pitch_hcqt, pitch_chromanet, pitch_crop_fn]):
        log_message("Render request requires pitch; initializing pitch models.")
        await initialize_pitch_models()

    log_message("Queueing render request | audio=%s | output=%s" % (audio_file, output_path))
    return await processing_queue.enqueue(
        _render_ground_truth_plan_logic,
        audio_file=audio_file,
        graph_spec=graph_spec,
        poison_graph_spec=poison_graph_spec,
        output_path=output_path,
        config_dir=config_dir,
        separation_cache_dir=separation_cache_dir,
        max_audio_seconds=max_audio_seconds,
        overwrite=overwrite,
        baseline_output_path=baseline_output_path,
        source_copy_path=source_copy_path,
    )


@server.tool()
def return_audio(
    audio_file: str,
    message: str,
) -> dict:
    """Return the final edited audio and corresponding message of edits performed."""
    return {"audio_path": audio_file, "message": message}


if __name__ == "__main__":
    try:
        server.run()
    finally:
        with open(log_file, "a") as f:
            f.write(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Shutting down Audio Processing Server...\n"
            )

        try:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(processing_queue.stop())
            except RuntimeError:
                asyncio.run(processing_queue.stop())
        except Exception as e:
            with open(log_file, "a") as f:
                f.write(
                    f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error during cleanup: {e}\n"
                )
