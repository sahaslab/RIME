import numpy as np
import torch
import torchaudio
from typing import Optional, Union, Callable, Any, Dict, Annotated
import time
import sys
import os
import asyncio
import tomllib
import re
from pydantic import Field

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

os.environ["TORCH_HOME"] = "/dartfs/rc/lab/S/SinghN/noah/.cache/torch"
torch.hub.set_dir(os.environ["TORCH_HOME"])

from mcp.server.fastmcp import FastMCP

from libraries.effects import (
    apply_chorus,
    apply_phaser,
    apply_distortion,
    apply_reverb,
    apply_delay,
    apply_compressor,
    apply_limiter,
    apply_deesser,
)
from libraries.mixing import (
    apply_gain,
    apply_highpass,
    apply_lowpass,
    apply_highshelf,
    apply_lowshelf,
    apply_peakfilter,
    apply_noisegate,
    normalize_peak,
    apply_fade_in_out,
    apply_pan,
    mix_stem_with_residual,
)
from libraries.separation import separate

from libraries.pitch import autotune, generate_harmony, apply_pitch_shift
from skey.key_detection import load_checkpoint, load_model_components
from audio_queue import AudioProcessingQueue

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
separation_device = torch.device("cpu")
processing_queue = AudioProcessingQueue()
log_file = "server_status.log"
DEFAULT_SERVER_CONFIG_PATH = os.environ.get(
    "POST_MASTER_SERVER_CONFIG",
    "/dartfs-hpc/rc/home/t/f00814t/lab/projects/RIME/configs/ground_truth/server/server.toml",
)
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
EDIT_TOOL_ALIASES = (
    # ("pitch shift", "apply_pitch_shift_effect"),
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
PLANNING_INPUT_TYPES = {"mixture", "single-source", "unknown"}
CONJUNCTION_ONLY_RE = re.compile(r"^\s*(?:,|and|then|plus|after that|afterwards)?\s*$")
DIRECT_OBJECT_STOP_WORDS = {
    "using",
    "with",
    "while",
    "then",
    "after",
    "before",
    "but",
    "so",
    "because",
    "if",
    "when",
    "until",
}
server_config: Dict[str, Any] = {
    "separation_backend": "demucs",
    "demucs_model": "htdemucs_6s",
    "sam_model": "facebook/sam-audio-large",
}
PEDALBOARD_PARAM_SPECS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "apply_chorus_effect": {
        "rate_hz": {"minimum": 0.0, "maximum": 100.0},
        "depth": {"minimum": 0.0, "maximum": 1.0},
        "centre_delay_ms": {"minimum": 0.0, "maximum": 100.0},
        "feedback": {"minimum": 0.0, "maximum": 1.0},
        "mix": {"minimum": 0.0, "maximum": 1.0},
    },
    "apply_phaser_effect": {
        "rate_hz": {"minimum": 0.0, "maximum": 100.0},
        "depth": {"minimum": 0.0, "maximum": 1.0},
        "centre_frequency_hz": {"minimum": 20.0, "maximum": 20000.0},
        "feedback": {"minimum": 0.0, "maximum": 1.0},
        "mix": {"minimum": 0.0, "maximum": 1.0},
    },
    "apply_distortion_effect": {
        "drive_db": {"minimum": 0.0, "maximum": 60.0},
    },
    # "apply_pitch_shift_effect": {
    #     "semitones": {"minimum": -48.0, "maximum": 48.0},
    # },
    "apply_reverb_effect": {
        "room_size": {"minimum": 0.0, "maximum": 1.0},
        "damping": {"minimum": 0.0, "maximum": 1.0},
        "wet_level": {"minimum": 0.0, "maximum": 1.0},
        "dry_level": {"minimum": 0.0, "maximum": 1.0},
        "width": {"minimum": 0.0, "maximum": 1.0},
        "freeze_mode": {"minimum": 0.0, "maximum": 1.0},
    },
    "apply_delay_effect": {
        "delay_seconds": {"minimum": 0.0, "maximum": 10.0},
        "feedback": {"minimum": 0.0, "maximum": 1.0},
        "mix": {"minimum": 0.0, "maximum": 1.0},
    },
    "apply_compressor_effect": {
        "threshold_db": {"minimum": -120.0, "maximum": 0.0},
        "ratio": {"minimum": 1.0, "maximum": 25.0},
        "attack_ms": {"minimum": 0.01, "maximum": 500.0},
        "release_ms": {"minimum": 1.0, "maximum": 5000.0},
    },
    "apply_limiter_effect": {
        "threshold_db": {"minimum": -120.0, "maximum": 0.0},
        "release_ms": {"minimum": 1.0, "maximum": 5000.0},
    },
    "apply_deesser_tool": {
        "ess_highpass_hz": {"minimum": 20.0, "maximum": 20000.0},
        "ess_lowpass_hz": {"minimum": 20.0, "maximum": 20000.0},
        "threshold_db": {"minimum": -120.0, "maximum": 0.0},
        "ratio": {"minimum": 1.0, "maximum": 25.0},
        "attack_ms": {"minimum": 0.01, "maximum": 500.0},
        "release_ms": {"minimum": 1.0, "maximum": 5000.0},
        "relative_threshold_db": {"minimum": -60.0, "maximum": 24.0},
        "max_reduction_db": {"minimum": 0.0, "maximum": 60.0},
        "filter_order": {"minimum": 1, "maximum": 12},
    },
    "apply_gain_tool": {
        "gain_db": {"minimum": -60.0, "maximum": 60.0},
    },
    "apply_highpass_filter_tool": {
        "cutoff_frequency_hz": {"minimum": 20.0, "maximum": 20000.0},
    },
    "apply_lowpass_filter_tool": {
        "cutoff_frequency_hz": {"minimum": 20.0, "maximum": 20000.0},
    },
    "apply_highshelf_filter_tool": {
        "cutoff_frequency_hz": {"minimum": 20.0, "maximum": 20000.0},
        "gain_db": {"minimum": -24.0, "maximum": 24.0},
        "q": {"minimum": 0.1, "maximum": 10.0},
    },
    "apply_lowshelf_filter_tool": {
        "cutoff_frequency_hz": {"minimum": 20.0, "maximum": 20000.0},
        "gain_db": {"minimum": -24.0, "maximum": 24.0},
        "q": {"minimum": 0.1, "maximum": 10.0},
    },
    "apply_peak_filter_tool": {
        "cutoff_frequency_hz": {"minimum": 20.0, "maximum": 20000.0},
        "gain_db": {"minimum": -24.0, "maximum": 24.0},
        "q": {"minimum": 0.1, "maximum": 10.0},
    },
    "apply_noisegate_tool": {
        "threshold_db": {"minimum": -120.0, "maximum": 0.0},
        "ratio": {"minimum": 1.0, "maximum": 25.0},
        "attack_ms": {"minimum": 0.01, "maximum": 500.0},
        "release_ms": {"minimum": 1.0, "maximum": 5000.0},
    },
    "normalize_peak_tool": {
        "target_peak": {"minimum": 0.0, "maximum": 1.0, "exclusive_minimum": True},
    },
    "apply_fade_in_out_tool": {
        "fade_in_seconds": {"minimum": 0.0},
        "fade_out_seconds": {"minimum": 0.0},
    },
    "apply_pan_tool": {
        "pan": {"minimum": -1.0, "maximum": 1.0},
    },
    "apply_harmony": {
        "semitones": {"minimum": -48.0, "maximum": 48.0},
    },
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def load_server_config(config_path: str = DEFAULT_SERVER_CONFIG_PATH) -> Dict[str, Any]:
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
    with open(log_file, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")


def load_audio(file_path: str) -> tuple[np.ndarray, int]:
    """Load audio from file and return audio data and sample rate."""
    try:
        audio_tensor, sr = torchaudio.load(file_path)
        audio = audio_tensor.numpy()
        return audio, sr
    except Exception as e:
        raise ValueError(f"Error loading audio file: {str(e)}")


def load_audio_as_tensor(file_path: str) -> tuple[torch.Tensor, int]:
    """Load audio from file and return audio tensor and sample rate."""
    try:
        audio_tensor, sr = torchaudio.load(file_path)
        return audio_tensor, sr
    except Exception as e:
        raise ValueError(f"Error loading audio file: {str(e)}")


def _field_for_param(tool_name: str, param_name: str) -> Any:
    spec = PEDALBOARD_PARAM_SPECS.get(tool_name, {}).get(param_name, {})
    field_kwargs: Dict[str, Any] = {}

    minimum = spec.get("minimum")
    maximum = spec.get("maximum")
    if minimum is not None:
        if spec.get("exclusive_minimum"):
            field_kwargs["gt"] = minimum
        else:
            field_kwargs["ge"] = minimum
    if maximum is not None:
        if spec.get("exclusive_maximum"):
            field_kwargs["lt"] = maximum
        else:
            field_kwargs["le"] = maximum

    return Field(**field_kwargs)


def _build_validation_error(
    tool_name: str,
    *,
    message: str,
    invalid_params: list[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "error": message,
        "error_type": "invalid_tool_arguments",
        "tool_name": tool_name,
        "invalid_params": invalid_params,
        "retry_same_assignment": True,
    }


def _build_runtime_error(
    tool_name: str,
    *,
    message: str,
    error_type: str = "tool_execution_failed",
) -> Dict[str, Any]:
    return {
        "error": message,
        "error_type": error_type,
        "tool_name": tool_name,
        "retry_same_assignment": True,
    }


def _load_audio_sample_rate(audio_file: str) -> Optional[int]:
    try:
        metadata = torchaudio.info(audio_file)
        sample_rate = getattr(metadata, "sample_rate", None)
        if isinstance(sample_rate, int) and sample_rate > 0:
            return sample_rate
    except Exception:
        pass

    try:
        _, sample_rate = load_audio(audio_file)
        return sample_rate
    except Exception:
        return None


def _validate_tool_arguments(
    tool_name: str,
    *,
    audio_file: Optional[str],
    arguments: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    invalid_params: list[Dict[str, Any]] = []
    sample_rate: Optional[int] = None
    nyquist: Optional[float] = None

    for param_name, spec in PEDALBOARD_PARAM_SPECS.get(tool_name, {}).items():
        if param_name not in arguments:
            continue
        value = arguments[param_name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            invalid_params.append(
                {
                    "name": param_name,
                    "provided": value,
                    "reason": "not_numeric",
                }
            )
            continue

        minimum = spec.get("minimum")
        if minimum is not None:
            if spec.get("exclusive_minimum"):
                if not value > minimum:
                    invalid_params.append(
                        {
                            "name": param_name,
                            "provided": value,
                            "reason": "below_or_equal_minimum",
                            "minimum": minimum,
                        }
                    )
                    continue
            elif value < minimum:
                invalid_params.append(
                    {
                        "name": param_name,
                        "provided": value,
                        "reason": "below_minimum",
                        "minimum": minimum,
                    }
                )
                continue

        maximum = spec.get("maximum")
        if maximum is not None:
            if spec.get("exclusive_maximum"):
                if not value < maximum:
                    invalid_params.append(
                        {
                            "name": param_name,
                            "provided": value,
                            "reason": "above_or_equal_maximum",
                            "maximum": maximum,
                        }
                    )
                    continue
            elif value > maximum:
                invalid_params.append(
                    {
                        "name": param_name,
                        "provided": value,
                        "reason": "above_maximum",
                        "maximum": maximum,
                    }
                )
                continue

    frequency_params = {"cutoff_frequency_hz", "centre_frequency_hz", "ess_lowpass_hz"}
    if any(name in arguments for name in frequency_params) and audio_file:
        sample_rate = _load_audio_sample_rate(audio_file)
        if sample_rate:
            nyquist = sample_rate / 2.0

    if nyquist is not None:
        for param_name in frequency_params:
            value = arguments.get(param_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= nyquist:
                invalid_params.append(
                    {
                        "name": param_name,
                        "provided": value,
                        "reason": "must_be_below_nyquist",
                        "maximum": nyquist,
                    }
                )

    if tool_name == "apply_deesser_tool":
        low = arguments.get("ess_highpass_hz")
        high = arguments.get("ess_lowpass_hz")
        if (
            isinstance(low, (int, float))
            and not isinstance(low, bool)
            and isinstance(high, (int, float))
            and not isinstance(high, bool)
            and low >= high
        ):
            invalid_params.extend(
                [
                    {
                        "name": "ess_highpass_hz",
                        "provided": low,
                        "reason": "must_be_lower_than_ess_lowpass_hz",
                        "other_param": "ess_lowpass_hz",
                        "other_value": high,
                    },
                    {
                        "name": "ess_lowpass_hz",
                        "provided": high,
                        "reason": "must_be_higher_than_ess_highpass_hz",
                        "other_param": "ess_highpass_hz",
                        "other_value": low,
                    },
                ]
            )

    if invalid_params:
        return _build_validation_error(
            tool_name,
            message=f"{tool_name} received invalid argument values.",
            invalid_params=invalid_params,
        )

    return None


def _run_validated_simple_effect(
    *,
    tool_name: str,
    audio_file: str,
    output_path: str,
    processor: Callable[..., Union[np.ndarray, torch.Tensor]],
    processor_args: tuple = (),
    processor_kwargs: Optional[dict] = None,
    validation_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    validation_error = _validate_tool_arguments(
        tool_name,
        audio_file=audio_file,
        arguments=validation_args or {},
    )
    if validation_error is not None:
        return validation_error

    try:
        return _process_simple_effect(
            audio_file=audio_file,
            output_path=output_path,
            processor=processor,
            processor_args=processor_args,
            processor_kwargs=processor_kwargs,
        )
    except Exception as exc:
        return _build_runtime_error(tool_name, message=str(exc))


def save_audio(
    audio_data: Union[np.ndarray, torch.Tensor],
    sample_rate: int,
    output_path: str,
) -> str:
    """Save audio arrays or tensors to disk. output_path is required."""
    if not output_path or not isinstance(output_path, str):
        raise ValueError("output_path must be a non-empty string")

    # Convert numpy to tensor if necessary
    if isinstance(audio_data, np.ndarray):
        audio_tensor = torch.from_numpy(audio_data)
    elif isinstance(audio_data, torch.Tensor):
        audio_tensor = audio_data.detach().cpu()
    else:
        raise ValueError("audio_data must be a numpy array or torch tensor")

    # torchaudio.save requires a 2D tensor [channels, frames]
    if audio_tensor.dim() == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    torchaudio.save(output_path, audio_tensor, sample_rate)
    return output_path


def _process_simple_effect(
    *,
    audio_file: str,
    output_path: str,
    processor: Callable[..., Union[np.ndarray, torch.Tensor]],
    processor_args: tuple = (),
    processor_kwargs: Optional[dict] = None,
) -> Dict[str, Any]:
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


def _guess_input_type(user_request: str, proposed_input_type: Optional[str] = None) -> str:
    if isinstance(proposed_input_type, str):
        normalized = proposed_input_type.strip().lower()
        if normalized in PLANNING_INPUT_TYPES:
            return normalized

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


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_INVALID_PARAM_VALUE = object()


def _normalize_param_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        normalized = _normalize_whitespace(value)
        return normalized if normalized else _INVALID_PARAM_VALUE
    if isinstance(value, list):
        normalized_items = []
        for item in value:
            normalized_item = _normalize_param_value(item)
            if normalized_item is _INVALID_PARAM_VALUE:
                continue
            normalized_items.append(normalized_item)
        return normalized_items
    if isinstance(value, dict):
        normalized_dict: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            normalized_key = _normalize_whitespace(key)
            if not normalized_key:
                continue
            normalized_item = _normalize_param_value(item)
            if normalized_item is _INVALID_PARAM_VALUE:
                continue
            normalized_dict[normalized_key] = normalized_item
        return normalized_dict
    return _INVALID_PARAM_VALUE


def _normalize_assignment_params(params: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(params, dict):
        return None

    normalized_params: Dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str):
            continue
        normalized_key = _normalize_whitespace(key)
        if not normalized_key:
            continue
        normalized_value = _normalize_param_value(value)
        if normalized_value is _INVALID_PARAM_VALUE:
            continue
        normalized_params[normalized_key] = normalized_value

    return normalized_params or None


def _map_effect_phrase_to_tool(effect_phrase: Optional[str]) -> Optional[str]:
    if not isinstance(effect_phrase, str):
        return None
    effect_text = effect_phrase.lower()
    for alias, tool_name in EDIT_TOOL_ALIASES:
        if alias in effect_text:
            return tool_name
    return None


def _resolve_tool_name_against_available_tools(
    tool_name: Optional[str],
    available_tool_names: Optional[list[str]],
) -> Optional[str]:
    if not isinstance(tool_name, str) or not tool_name.strip():
        return None
    if not available_tool_names:
        return tool_name.strip()

    stripped_name = tool_name.strip()
    return stripped_name if stripped_name in available_tool_names else None


def _sentence_end(text: str, start: int) -> int:
    sentence_end = len(text)
    for delimiter in ".;!?":
        candidate_end = text.find(delimiter, start)
        if candidate_end != -1:
            sentence_end = min(sentence_end, candidate_end)
    return sentence_end


def _clean_source_phrase(text: str) -> Optional[str]:
    cleaned = _normalize_whitespace(text.strip(" ,.;:!?"))
    cleaned = re.sub(r"^(?:to|on|onto|for|in)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^(?:the|this|that|these|those|my|our|your|their|a|an)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+(?:and|then|plus)\s*$", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip(" ,.;:!?")
    if cleaned.lower() in {"and", "then", "plus"}:
        return None
    return cleaned or None


def _extract_prepositional_target(
    request_text: str,
    effect_end: int,
    next_effect_start: int,
) -> Optional[str]:
    text_after_effect = request_text[effect_end:]
    match = re.search(r"\b(?:to|on|onto|for|in)\b", text_after_effect)
    if match is None:
        return None

    target_start = effect_end + match.end()
    target_end = min(_sentence_end(request_text, effect_end), next_effect_start)
    raw_target = request_text[target_start:target_end]
    return _clean_source_phrase(raw_target)


def _extract_direct_object_target(
    request_text: str,
    effect_end: int,
    next_effect_start: int,
) -> Optional[str]:
    target_end = min(_sentence_end(request_text, effect_end), next_effect_start)
    remainder = request_text[effect_end:target_end]
    cleaned = _normalize_whitespace(remainder)
    if not cleaned:
        return None

    cleaned = re.sub(
        r"^(?:to|on|onto|for|in)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    if not cleaned:
        return None

    tokens = cleaned.split()
    collected: list[str] = []
    for token in tokens:
        token_lower = token.lower().strip(",.;:!?")
        if token_lower in {"and", "plus"} and not collected:
            break
        if token_lower in DIRECT_OBJECT_STOP_WORDS:
            break
        if token_lower in {"and", "plus"} and collected:
            break
        collected.append(token)

    if not collected:
        return None

    return _clean_source_phrase(" ".join(collected))


def _extract_requested_assignments(user_request: str) -> list[Dict[str, Any]]:
    request_lower = user_request.lower()
    effect_mentions: list[Dict[str, Any]] = []

    for alias, tool_name in EDIT_TOOL_ALIASES:
        for match in re.finditer(re.escape(alias), request_lower):
            effect_mentions.append(
                {
                    "position": match.start(),
                    "match_end": match.end(),
                    "effect_phrase": match.group(0),
                    "tool_name": tool_name,
                }
            )

    effect_mentions.sort(key=lambda item: int(item["position"]))
    if not effect_mentions:
        return []

    assignments: list[Dict[str, Any]] = []
    for index, mention in enumerate(effect_mentions):
        next_effect_start = len(request_lower)
        if index + 1 < len(effect_mentions):
            next_effect_start = int(effect_mentions[index + 1]["position"])

        source_phrase = _extract_prepositional_target(
            user_request,
            int(mention["match_end"]),
            next_effect_start,
        )
        if source_phrase is None:
            source_phrase = _extract_direct_object_target(
                user_request,
                int(mention["match_end"]),
                next_effect_start,
            )

        assignments.append(
            {
                "position": int(mention["position"]),
                "effect_phrase": str(mention["effect_phrase"]),
                "tool_name": str(mention["tool_name"]),
                "source_phrase": source_phrase,
            }
        )

    for index, assignment in enumerate(assignments):
        if assignment["source_phrase"] is not None:
            continue
        if index + 1 >= len(assignments):
            continue
        between = request_lower[
            int(effect_mentions[index]["match_end"]) : int(effect_mentions[index + 1]["position"])
        ]
        next_source = assignments[index + 1]["source_phrase"]
        if next_source is not None and CONJUNCTION_ONLY_RE.match(between):
            assignment["source_phrase"] = next_source

    for index, assignment in enumerate(assignments):
        if assignment["source_phrase"] is not None:
            continue
        if index == 0:
            continue
        previous_source = assignments[index - 1]["source_phrase"]
        between = request_lower[
            int(effect_mentions[index - 1]["match_end"]) : int(effect_mentions[index]["position"])
        ]
        if previous_source is not None and CONJUNCTION_ONLY_RE.match(between):
            assignment["source_phrase"] = previous_source

    deduped: list[Dict[str, Any]] = []
    seen_keys = set()
    for assignment in assignments:
        dedupe_key = (
            assignment["position"],
            assignment["tool_name"],
            assignment["source_phrase"],
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped.append(assignment)

    return deduped


def _normalize_assignments(
    assignments: Optional[list[Dict[str, Any]]],
    user_request: str,
    available_tool_names: Optional[list[str]] = None,
) -> list[Dict[str, Any]]:
    normalized: list[Dict[str, Any]] = []
    if isinstance(assignments, list):
        for assignment in assignments:
            if not isinstance(assignment, dict):
                continue

            effect_phrase = assignment.get("effect_phrase")
            tool_name = assignment.get("tool_name")
            source_phrase = assignment.get("source_phrase")
            params = _normalize_assignment_params(assignment.get("params"))

            if not isinstance(effect_phrase, str) or not effect_phrase.strip():
                continue
            effect_phrase = _normalize_whitespace(effect_phrase)

            if not isinstance(tool_name, str) or not tool_name.strip():
                resolved_tool_name = _resolve_tool_name_against_available_tools(
                    _map_effect_phrase_to_tool(effect_phrase),
                    available_tool_names,
                )
            else:
                resolved_tool_name = _resolve_tool_name_against_available_tools(
                    tool_name, available_tool_names
                )
            if not isinstance(resolved_tool_name, str) or not resolved_tool_name.strip():
                continue

            entry: Dict[str, Any] = {
                "effect_phrase": effect_phrase,
                "tool_name": resolved_tool_name.strip(),
            }

            if isinstance(source_phrase, str):
                cleaned_source = _clean_source_phrase(source_phrase)
                if cleaned_source:
                    entry["source_phrase"] = cleaned_source
            if params:
                entry["params"] = params

            normalized.append(entry)

    if normalized:
        return normalized

    fallback_assignments = _extract_requested_assignments(user_request)
    fallback_normalized: list[Dict[str, Any]] = []
    for assignment in fallback_assignments:
        effect_phrase = assignment.get("effect_phrase")
        if not isinstance(effect_phrase, str) or not effect_phrase.strip():
            continue
        resolved_tool_name = _resolve_tool_name_against_available_tools(
            assignment.get("tool_name"),
            available_tool_names,
        )
        if not isinstance(resolved_tool_name, str) or not resolved_tool_name.strip():
            continue

        entry: Dict[str, Any] = {
            "effect_phrase": effect_phrase.strip(),
            "tool_name": resolved_tool_name.strip(),
        }
        source_phrase = assignment.get("source_phrase")
        if isinstance(source_phrase, str):
            cleaned_source = _clean_source_phrase(source_phrase)
            if cleaned_source:
                entry["source_phrase"] = cleaned_source
        fallback_normalized.append(entry)

    return fallback_normalized


def _group_operations_by_target(
    operations: list[Dict[str, str]],
) -> list[Dict[str, Any]]:
    groups: list[Dict[str, Any]] = []
    for operation in operations:
        target = operation.get("source_phrase")
        if groups and groups[-1]["target"] == target:
            groups[-1]["operations"].append(operation)
            continue
        groups.append({"target": target, "operations": [operation]})
    return groups


def _build_plan_text(
    user_request: str,
    *,
    input_type: str,
    assignments: list[Dict[str, str]],
) -> str:
    requested_edits = ", ".join(_collect_requested_edits(user_request))
    operation_groups = _group_operations_by_target(assignments)

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
                    f"{step_number}. Keep the binding exact: only process {group_target} for this branch. Likely tool: separate_audio(audio_file=CURRENT_AUDIO, description=\"{group_target}\", return_both=true)."
                )
                step_number += 1
                for operation in group_operations:
                    steps.append(
                        f"{step_number}. Likely tool: {operation['tool_name']}(audio_file=STEM_PATH, output_path=REQUEST_DIR + \"/...\") so {operation['effect_phrase']} stays attached to {group_target}."
                    )
                    step_number += 1
                steps.append(
                    f"{step_number}. Likely tool: mix_sources(processed_stem_file=PROCESSED_STEM_PATH, residual_file=RESIDUAL_PATH, output_path=REQUEST_DIR + \"/...\") to rebuild the full mix before the next edit."
                )
                step_number += 1
            else:
                for operation in group_operations:
                    steps.append(
                        f"{step_number}. Likely tool: {operation['tool_name']}(audio_file=CURRENT_AUDIO, output_path=REQUEST_DIR + \"/...\") directly on the current audio."
                    )
                    step_number += 1

        steps.append(
            f"{step_number}. Likely tool: return_audio(audio_file=CURRENT_AUDIO, message=\"...\") once the remixed result satisfies the request."
        )
    elif input_type == "unknown":
        steps.extend(
            [
                "2. Do not assume this is a mixture just because the request mentions a source phrase.",
                "3. Preserve the planned source-to-effect bindings exactly unless listening clearly proves they were misread.",
                "4. If listening confirms a full mix, handle each source-targeted assignment sequentially: separate_audio for that source phrase, apply the matching effect tool to STEM_PATH, then mix_sources before moving to the next assignment.",
                "5. If listening confirms an isolated source, apply the matching effect tools directly to CURRENT_AUDIO in sequence without separation.",
                "6. If multiple assignments target different sources, finish one separation/process/mix cycle before starting the next one.",
                "7. After each edit, re-listen and deviate from this plan whenever the latest audio suggests a better sequence or parameter choice.",
                "8. Finish with return_audio once the strongest result matches the request.",
            ]
        )
    else:
        step_number = 2
        if operation_groups:
            for group in operation_groups:
                for operation in group["operations"]:
                    target_phrase = operation.get("source_phrase")
                    if target_phrase:
                        steps.append(
                            f"{step_number}. Preserve the binding {operation['effect_phrase']} -> {target_phrase} and likely use {operation['tool_name']}(audio_file=CURRENT_AUDIO, output_path=REQUEST_DIR + \"/...\") directly on the isolated source."
                        )
                    else:
                        steps.append(
                            f"{step_number}. Likely tool: {operation['tool_name']}(audio_file=CURRENT_AUDIO, output_path=REQUEST_DIR + \"/...\") directly on the isolated source."
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


def _build_fallback_plan_payload(
    user_request: str,
    audio_file: Optional[str],
    input_type: Optional[str],
    assignments: Optional[list[Dict[str, Any]]],
    plan: Optional[str],
    available_tool_names: Optional[list[str]] = None,
) -> dict:
    normalized_input_type = _guess_input_type(user_request, input_type)
    normalized_assignments = _normalize_assignments(
        assignments, user_request, available_tool_names
    )
    normalized_plan = (
        plan.strip()
        if isinstance(plan, str) and plan.strip()
        else _build_plan_text(
            user_request,
            input_type=normalized_input_type,
            assignments=normalized_assignments,
        )
    )

    return {
        "plan": normalized_plan,
        "assignments": normalized_assignments,
        "input_type": normalized_input_type,
        "user_request": user_request,
        "audio_file": audio_file,
        "advisory_only": True,
    }


@server.tool()
def plan_edit_sequence(
    user_request: str,
    audio_file: Optional[str] = None,
    input_type: Optional[str] = None,
    assignments: Optional[list[Dict[str, Any]]] = None,
    plan: Optional[str] = None,
    available_tool_names: Optional[list[str]] = None,
) -> dict:
    """Create an advisory edit plan. Prefer passing input_type, assignments, and plan generated from the user request; assignments should use the user's exact source phrases when possible and may include advisory params."""
    return _build_fallback_plan_payload(
        user_request=user_request,
        audio_file=audio_file,
        input_type=input_type,
        assignments=assignments,
        plan=plan,
        available_tool_names=available_tool_names,
    )


# ============================================================================
# SEPARATION TOOLS
# ============================================================================


async def initialize_separation_models():
    global separation_model, separation_processor
    if separation_model is not None:
        return

    if not processing_queue.is_running:
        await processing_queue.start()

    if separation_model is not None:
        return

    separation_backend = str(server_config["separation_backend"]).lower()

    if separation_backend == "demucs":
        try:
            from demucs.pretrained import get_model
        except ImportError as exc:
            raise ImportError(
                "Demucs is not installed. Install the `demucs` package to use separation."
            ) from exc

        model_name = str(server_config["demucs_model"])
        separation_model = get_model(name=model_name).to(separation_device).eval()
        separation_processor = None
        log_message(f"Separation model initialized with Demucs ({model_name}).")
        return

    if separation_backend == "sam_audio":
        try:
            from sam_audio import SAMAudio, SAMAudioProcessor
            from sam_audio.model.config import SAMAudioConfig
        except ImportError as exc:
            raise ImportError(
                "SAM Audio is not installed. Install the `sam_audio` package to use separation."
            ) from exc

        cache_dir = "/dartfs/rc/lab/S/SinghN/noah/.cache/huggingface/hub"
        cfg = SAMAudioConfig(visual_ranker=None)
        model_name = str(server_config["sam_model"])
        separation_model = (
            SAMAudio.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype="auto",
                device_map="auto",
                use_safetensors=True,
                cache_dir=cache_dir,
                config=cfg,
            )
            .to(device)
            .eval()
        )
        separation_processor = SAMAudioProcessor.from_pretrained(model_name)
        log_message(f"Separation model initialized with SAM Audio ({model_name}).")
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
        separation_device,
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
    rate_hz: Annotated[float, _field_for_param("apply_chorus_effect", "rate_hz")],
    depth: Annotated[float, _field_for_param("apply_chorus_effect", "depth")],
    centre_delay_ms: Annotated[
        float, _field_for_param("apply_chorus_effect", "centre_delay_ms")
    ],
    feedback: Annotated[float, _field_for_param("apply_chorus_effect", "feedback")],
    mix: Annotated[float, _field_for_param("apply_chorus_effect", "mix")],
) -> dict:
    """Apply chorus effect to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_chorus_effect",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_chorus,
        processor_args=(rate_hz, depth, centre_delay_ms, feedback, mix),
        validation_args={
            "rate_hz": rate_hz,
            "depth": depth,
            "centre_delay_ms": centre_delay_ms,
            "feedback": feedback,
            "mix": mix,
        },
    )


@server.tool()
def apply_phaser_effect(
    audio_file: str,
    output_path: str,
    rate_hz: Annotated[float, _field_for_param("apply_phaser_effect", "rate_hz")],
    depth: Annotated[float, _field_for_param("apply_phaser_effect", "depth")],
    centre_frequency_hz: Annotated[
        float, _field_for_param("apply_phaser_effect", "centre_frequency_hz")
    ],
    feedback: Annotated[float, _field_for_param("apply_phaser_effect", "feedback")],
    mix: Annotated[float, _field_for_param("apply_phaser_effect", "mix")],
) -> dict:
    """Apply phaser effect to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_phaser_effect",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_phaser,
        processor_args=(rate_hz, depth, centre_frequency_hz, feedback, mix),
        validation_args={
            "rate_hz": rate_hz,
            "depth": depth,
            "centre_frequency_hz": centre_frequency_hz,
            "feedback": feedback,
            "mix": mix,
        },
    )


@server.tool()
def apply_distortion_effect(
    audio_file: str,
    output_path: str,
    drive_db: Annotated[
        float, _field_for_param("apply_distortion_effect", "drive_db")
    ],
) -> dict:
    """Apply distortion effect to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_distortion_effect",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_distortion,
        processor_args=(drive_db,),
        validation_args={"drive_db": drive_db},
    )


# @server.tool()
# def apply_pitch_shift_effect(
#     audio_file: str,
#     output_path: str,
#     semitones: Annotated[
#         float, _field_for_param("apply_pitch_shift_effect", "semitones")
#     ],
# ) -> dict:
#     """Apply a constant pitch shift to audio."""
#     return _run_validated_simple_effect(
#         tool_name="apply_pitch_shift_effect",
#         audio_file=audio_file,
#         output_path=output_path,
#         processor=apply_pitch_shift,
#         processor_args=(semitones,),
#         validation_args={"semitones": semitones},
#     )


@server.tool()
def apply_reverb_effect(
    audio_file: str,
    output_path: str,
    room_size: Annotated[float, _field_for_param("apply_reverb_effect", "room_size")],
    damping: Annotated[float, _field_for_param("apply_reverb_effect", "damping")],
    wet_level: Annotated[float, _field_for_param("apply_reverb_effect", "wet_level")],
    dry_level: Annotated[float, _field_for_param("apply_reverb_effect", "dry_level")],
    width: Annotated[float, _field_for_param("apply_reverb_effect", "width")],
    freeze_mode: Annotated[
        float, _field_for_param("apply_reverb_effect", "freeze_mode")
    ],
) -> dict:
    """Apply reverb effect to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_reverb_effect",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_reverb,
        processor_args=(room_size, damping, wet_level, dry_level, width, freeze_mode),
        validation_args={
            "room_size": room_size,
            "damping": damping,
            "wet_level": wet_level,
            "dry_level": dry_level,
            "width": width,
            "freeze_mode": freeze_mode,
        },
    )


@server.tool()
def apply_delay_effect(
    audio_file: str,
    output_path: str,
    delay_seconds: Annotated[
        float, _field_for_param("apply_delay_effect", "delay_seconds")
    ],
    feedback: Annotated[float, _field_for_param("apply_delay_effect", "feedback")],
    mix: Annotated[float, _field_for_param("apply_delay_effect", "mix")],
) -> dict:
    """Apply delay effect to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_delay_effect",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_delay,
        processor_args=(delay_seconds, feedback, mix),
        validation_args={
            "delay_seconds": delay_seconds,
            "feedback": feedback,
            "mix": mix,
        },
    )


@server.tool()
def apply_compressor_effect(
    audio_file: str,
    output_path: str,
    threshold_db: Annotated[
        float, _field_for_param("apply_compressor_effect", "threshold_db")
    ],
    ratio: Annotated[float, _field_for_param("apply_compressor_effect", "ratio")],
    attack_ms: Annotated[
        float, _field_for_param("apply_compressor_effect", "attack_ms")
    ],
    release_ms: Annotated[
        float, _field_for_param("apply_compressor_effect", "release_ms")
    ],
) -> dict:
    """Apply compressor to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_compressor_effect",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_compressor,
        processor_args=(threshold_db, ratio, attack_ms, release_ms),
        validation_args={
            "threshold_db": threshold_db,
            "ratio": ratio,
            "attack_ms": attack_ms,
            "release_ms": release_ms,
        },
    )


@server.tool()
def apply_limiter_effect(
    audio_file: str,
    output_path: str,
    threshold_db: Annotated[
        float, _field_for_param("apply_limiter_effect", "threshold_db")
    ],
    release_ms: Annotated[
        float, _field_for_param("apply_limiter_effect", "release_ms")
    ],
) -> dict:
    """Apply limiter to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_limiter_effect",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_limiter,
        processor_args=(threshold_db, release_ms),
        validation_args={
            "threshold_db": threshold_db,
            "release_ms": release_ms,
        },
    )


@server.tool()
def apply_deesser_tool(
    audio_file: str,
    output_path: str,
    ess_highpass_hz: Annotated[
        float, _field_for_param("apply_deesser_tool", "ess_highpass_hz")
    ],
    ess_lowpass_hz: Annotated[
        float, _field_for_param("apply_deesser_tool", "ess_lowpass_hz")
    ],
    threshold_db: Annotated[
        float, _field_for_param("apply_deesser_tool", "threshold_db")
    ],
    ratio: Annotated[float, _field_for_param("apply_deesser_tool", "ratio")],
    attack_ms: Annotated[float, _field_for_param("apply_deesser_tool", "attack_ms")],
    release_ms: Annotated[
        float, _field_for_param("apply_deesser_tool", "release_ms")
    ],
    relative_threshold_db: Annotated[
        float, _field_for_param("apply_deesser_tool", "relative_threshold_db")
    ],
    max_reduction_db: Annotated[
        float, _field_for_param("apply_deesser_tool", "max_reduction_db")
    ],
    filter_order: Annotated[int, _field_for_param("apply_deesser_tool", "filter_order")],
) -> dict:
    """Apply a simple split-band de-esser to reduce sibilance."""
    return _run_validated_simple_effect(
        tool_name="apply_deesser_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_deesser,
        processor_kwargs={
            "ess_low_hz": ess_highpass_hz,
            "ess_high_hz": ess_lowpass_hz,
            "threshold_db": threshold_db,
            "ratio": ratio,
            "attack_ms": attack_ms,
            "release_ms": release_ms,
            "relative_threshold_db": relative_threshold_db,
            "max_reduction_db": max_reduction_db,
            "filter_order": filter_order,
        },
        validation_args={
            "ess_highpass_hz": ess_highpass_hz,
            "ess_lowpass_hz": ess_lowpass_hz,
            "threshold_db": threshold_db,
            "ratio": ratio,
            "attack_ms": attack_ms,
            "release_ms": release_ms,
            "relative_threshold_db": relative_threshold_db,
            "max_reduction_db": max_reduction_db,
            "filter_order": filter_order,
        },
    )


# ============================================================================
# MIXING TOOLS
# ============================================================================


@server.tool()
def apply_gain_tool(
    audio_file: str,
    output_path: str,
    gain_db: Annotated[float, _field_for_param("apply_gain_tool", "gain_db")],
) -> dict:
    """Apply gain to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_gain_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_gain,
        processor_args=(gain_db,),
        validation_args={"gain_db": gain_db},
    )


@server.tool()
def apply_highpass_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: Annotated[
        float, _field_for_param("apply_highpass_filter_tool", "cutoff_frequency_hz")
    ],
) -> dict:
    """Apply highpass filter to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_highpass_filter_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_highpass,
        processor_args=(cutoff_frequency_hz,),
        validation_args={"cutoff_frequency_hz": cutoff_frequency_hz},
    )


@server.tool()
def apply_lowpass_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: Annotated[
        float, _field_for_param("apply_lowpass_filter_tool", "cutoff_frequency_hz")
    ],
) -> dict:
    """Apply lowpass filter to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_lowpass_filter_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_lowpass,
        processor_args=(cutoff_frequency_hz,),
        validation_args={"cutoff_frequency_hz": cutoff_frequency_hz},
    )


@server.tool()
def apply_highshelf_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: Annotated[
        float, _field_for_param("apply_highshelf_filter_tool", "cutoff_frequency_hz")
    ],
    gain_db: Annotated[
        float, _field_for_param("apply_highshelf_filter_tool", "gain_db")
    ],
    q: Annotated[float, _field_for_param("apply_highshelf_filter_tool", "q")],
) -> dict:
    """Apply high shelf filter to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_highshelf_filter_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_highshelf,
        processor_args=(cutoff_frequency_hz, gain_db, q),
        validation_args={
            "cutoff_frequency_hz": cutoff_frequency_hz,
            "gain_db": gain_db,
            "q": q,
        },
    )


@server.tool()
def apply_lowshelf_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: Annotated[
        float, _field_for_param("apply_lowshelf_filter_tool", "cutoff_frequency_hz")
    ],
    gain_db: Annotated[
        float, _field_for_param("apply_lowshelf_filter_tool", "gain_db")
    ],
    q: Annotated[float, _field_for_param("apply_lowshelf_filter_tool", "q")],
) -> dict:
    """Apply low shelf filter to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_lowshelf_filter_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_lowshelf,
        processor_args=(cutoff_frequency_hz, gain_db, q),
        validation_args={
            "cutoff_frequency_hz": cutoff_frequency_hz,
            "gain_db": gain_db,
            "q": q,
        },
    )


@server.tool()
def apply_peak_filter_tool(
    audio_file: str,
    output_path: str,
    cutoff_frequency_hz: Annotated[
        float, _field_for_param("apply_peak_filter_tool", "cutoff_frequency_hz")
    ],
    gain_db: Annotated[
        float, _field_for_param("apply_peak_filter_tool", "gain_db")
    ],
    q: Annotated[float, _field_for_param("apply_peak_filter_tool", "q")],
) -> dict:
    """Apply peak filter (parametric EQ) to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_peak_filter_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_peakfilter,
        processor_args=(cutoff_frequency_hz, gain_db, q),
        validation_args={
            "cutoff_frequency_hz": cutoff_frequency_hz,
            "gain_db": gain_db,
            "q": q,
        },
    )


@server.tool()
def apply_noisegate_tool(
    audio_file: str,
    output_path: str,
    threshold_db: Annotated[
        float, _field_for_param("apply_noisegate_tool", "threshold_db")
    ],
    ratio: Annotated[float, _field_for_param("apply_noisegate_tool", "ratio")],
    attack_ms: Annotated[float, _field_for_param("apply_noisegate_tool", "attack_ms")],
    release_ms: Annotated[
        float, _field_for_param("apply_noisegate_tool", "release_ms")
    ],
) -> dict:
    """Apply noise gate to audio."""
    return _run_validated_simple_effect(
        tool_name="apply_noisegate_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_noisegate,
        processor_args=(threshold_db, ratio, attack_ms, release_ms),
        validation_args={
            "threshold_db": threshold_db,
            "ratio": ratio,
            "attack_ms": attack_ms,
            "release_ms": release_ms,
        },
    )


@server.tool()
def normalize_peak_tool(
    audio_file: str,
    output_path: str,
    target_peak: Annotated[
        float, _field_for_param("normalize_peak_tool", "target_peak")
    ],
) -> dict:
    """Scale audio so the peak sample reaches the requested headroom target."""
    return _run_validated_simple_effect(
        tool_name="normalize_peak_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=normalize_peak,
        processor_args=(target_peak,),
        validation_args={"target_peak": target_peak},
    )


@server.tool()
def apply_fade_in_out_tool(
    audio_file: str,
    output_path: str,
    fade_in_seconds: Annotated[
        float, _field_for_param("apply_fade_in_out_tool", "fade_in_seconds")
    ],
    fade_out_seconds: Annotated[
        float, _field_for_param("apply_fade_in_out_tool", "fade_out_seconds")
    ],
) -> dict:
    """Apply linear fade-in and fade-out to the provided audio file."""
    return _run_validated_simple_effect(
        tool_name="apply_fade_in_out_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_fade_in_out,
        processor_args=(fade_in_seconds, fade_out_seconds),
        validation_args={
            "fade_in_seconds": fade_in_seconds,
            "fade_out_seconds": fade_out_seconds,
        },
    )


@server.tool()
def apply_pan_tool(
    audio_file: str,
    output_path: str,
    pan: Annotated[float, _field_for_param("apply_pan_tool", "pan")],
) -> dict:
    """Apply constant-power stereo panning. Negative is left, positive is right."""
    return _run_validated_simple_effect(
        tool_name="apply_pan_tool",
        audio_file=audio_file,
        output_path=output_path,
        processor=apply_pan,
        processor_args=(pan,),
        validation_args={"pan": pan},
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
    global pitch_hcqt, pitch_chromanet, pitch_crop_fn
    if pitch_hcqt is not None:
        return

    if not processing_queue.is_running:
        await processing_queue.start()

    if pitch_hcqt is not None:
        return
    ckpt = load_checkpoint("models/skey.pt")
    pitch_hcqt, pitch_chromanet, pitch_crop_fn = load_model_components(ckpt, device)
    log_message("Pitch models initialized.")


def _autotune_logic(
    audio_file: str,
    key: Optional[str],
    mode: Optional[str],
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
    key: Optional[str],
    mode: Optional[str],
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
    key: Optional[str] = None,
    mode: Optional[str] = None,
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
    semitones: Annotated[float, _field_for_param("apply_harmony", "semitones")],
    output_path: str,
    key: Optional[str] = None,
    mode: Optional[str] = None,
) -> dict:
    """Generate a harmony at a specified semitone distance; returns harmony-only."""
    validation_error = _validate_tool_arguments(
        "apply_harmony",
        audio_file=audio_file,
        arguments={"semitones": semitones},
    )
    if validation_error is not None:
        return validation_error
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