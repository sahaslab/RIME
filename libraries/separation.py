from __future__ import annotations
import os
import time
from libraries.separation_targets import normalize_demucs_6s_target
import torch
import torchaudio.functional as F
from typing import Any


DEFAULT_CHUNK_SECONDS = 15.0
DEFAULT_RERANKING_CANDIDATES = 1
DEFAULT_OOM_RERANKING_CANDIDATES = 1
DEFAULT_PREDICT_SPANS = False
DEFAULT_DEMUCS_OVERLAP = 0.5
DEFAULT_DEMUCS_PROGRESS = True
DEFAULT_DEMUCS_NUM_WORKERS = 8


def _canonicalize_description(description: str, available_sources: list[str]) -> str:
    normalized = normalize_demucs_6s_target(description)
    if normalized.canonical is None:
        raise ValueError(
            f"Could not map description '{description}' to Demucs sources {available_sources}"
        )
    if normalized.canonical not in available_sources:
        raise ValueError(
            f"Mapped description '{description}' to '{normalized.canonical}', "
            f"but available Demucs sources are {available_sources}"
        )
    return normalized.canonical


def _ensure_demucs_channels(audio: torch.Tensor) -> tuple[torch.Tensor, int]:
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.dim() != 2:
        raise ValueError(f"Expected audio with shape [channels, frames], got {tuple(audio.shape)}")

    original_channels = audio.shape[0]
    if original_channels == 1:
        audio = audio.repeat(2, 1)
    elif original_channels > 2:
        audio = audio[:2]

    return audio, original_channels


def _restore_original_channels(audio: torch.Tensor, original_channels: int) -> torch.Tensor:
    if original_channels <= 1:
        return audio[:1]
    return audio[:original_channels]


def _separate_with_sam(
    model,
    processor,
    device,
    audio: torch.Tensor,
    description: str,
    sample_rate: int | None,
):
    target_sample_rate = int(getattr(processor, "audio_sampling_rate", 48_000))
    working_audio = audio
    if sample_rate is not None and sample_rate != target_sample_rate:
        working_audio = F.resample(audio, sample_rate, target_sample_rate)

    settings = _resolve_settings(working_audio, target_sample_rate)
    chunk_samples = settings["chunk_samples"]
    if chunk_samples is None or working_audio.shape[-1] <= chunk_samples:
        _log(
            "separating without chunking | description=%s | samples=%d | reranking_candidates=%d | predict_spans=%s"
            % (
                description,
                working_audio.shape[-1],
                settings["reranking_candidates"],
                settings["predict_spans"],
            )
        )
        stem, residual = _separate_once(model, processor, device, working_audio, description, settings)
    else:
        stem_chunks: list[torch.Tensor] = []
        residual_chunks: list[torch.Tensor] = []
        total_samples = working_audio.shape[-1]
        total_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        _log(
            "separating in chunks | description=%s | total_samples=%d | chunk_samples=%d | chunks=%d | reranking_candidates=%d | predict_spans=%s"
            % (
                description,
                total_samples,
                chunk_samples,
                total_chunks,
                settings["reranking_candidates"],
                settings["predict_spans"],
            )
        )

        for chunk_index, start in enumerate(range(0, total_samples, chunk_samples), start=1):
            end = min(start + chunk_samples, total_samples)
            chunk = working_audio[..., start:end]
            chunk_started_at = time.perf_counter()
            _log(
                "chunk %d/%d | description=%s | sample_range=%d:%d"
                % (chunk_index, total_chunks, description, start, end)
            )
            chunk_stem, chunk_residual = _separate_once(model, processor, device, chunk, description, settings)
            stem_chunks.append(chunk_stem.detach().cpu())
            residual_chunks.append(chunk_residual.detach().cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _log(
                "chunk %d/%d done | description=%s | elapsed=%.1fs"
                % (chunk_index, total_chunks, description, time.perf_counter() - chunk_started_at)
            )

        stem = torch.cat(stem_chunks, dim=-1)
        residual = torch.cat(residual_chunks, dim=-1)

    if sample_rate is not None and sample_rate != target_sample_rate:
        stem = F.resample(stem, target_sample_rate, sample_rate)
        residual = F.resample(residual, target_sample_rate, sample_rate)
    return stem, residual


def _separate_once(model, processor, device, audio: torch.Tensor, description: str, settings: dict[str, int | bool | None]):
    model_input = processor(audios=[audio], descriptions=[description]).to(device)
    try:
        with torch.inference_mode():
            output = model.separate(
                model_input,
                predict_spans=bool(settings["predict_spans"]),
                reranking_candidates=int(settings["reranking_candidates"]),
            )
    except RuntimeError as error:
        if device.type != "cuda" or "out of memory" not in str(error).lower():
            raise
        _log(
            "OOM retry | description=%s | samples=%d | reranking_candidates=%d | predict_spans=%s"
            % (
                description,
                audio.shape[-1],
                int(settings["oom_reranking_candidates"]),
                False,
            )
        )
        del model_input
        torch.cuda.empty_cache()
        model_input = processor(audios=[audio], descriptions=[description]).to(device)
        with torch.inference_mode():
            output = model.separate(
                model_input,
                predict_spans=False,
                reranking_candidates=int(settings["oom_reranking_candidates"]),
            )
    stem = output.target[0].unsqueeze(0)
    residual = output.residual[0].unsqueeze(0)
    return stem, residual


def _separate_with_demucs(
    model,
    device,
    audio: torch.Tensor,
    description: str,
    sample_rate: int | None,
):
    source_estimates = separate_demucs_sources(model, device, audio, sample_rate)
    sources = list(source_estimates.keys())
    target_source = _canonicalize_description(description, sources)
    stem = source_estimates[target_source]
    residual_sources = [
        source_audio
        for source_name, source_audio in source_estimates.items()
        if source_name != target_source
    ]
    residual = torch.stack(residual_sources, dim=0).sum(dim=0)
    return stem, residual


def canonicalize_demucs_description(description: str, available_sources: list[str]) -> str:
    return _canonicalize_description(description, available_sources)


def separate_demucs_sources(
    model: Any,
    device: torch.device,
    audio: torch.Tensor,
    sample_rate: int | None,
) -> dict[str, torch.Tensor]:
    try:
        from demucs.apply import apply_model
    except ImportError as exc:
        raise ImportError(
            "Demucs is not installed. Install the `demucs` package to use the Demucs separation backend."
        ) from exc

    if sample_rate is None:
        raise ValueError("sample_rate is required for Demucs separation")

    demucs_audio, original_channels = _ensure_demucs_channels(audio.float().cpu())
    target_sample_rate = int(getattr(model, "samplerate", sample_rate))

    if sample_rate != target_sample_rate:
        demucs_audio = F.resample(demucs_audio, sample_rate, target_sample_rate)

    mixture = demucs_audio.unsqueeze(0).to(device)
    demucs_overlap = float(os.environ.get("DEMUCS_OVERLAP", str(DEFAULT_DEMUCS_OVERLAP)))
    demucs_num_workers = int(os.environ.get("DEMUCS_NUM_WORKERS", str(DEFAULT_DEMUCS_NUM_WORKERS)))
    demucs_progress = _env_flag("DEMUCS_PROGRESS", DEFAULT_DEMUCS_PROGRESS)
    _log(
        "demucs start | samples=%d | sr=%d | target_sr=%d | device=%s | overlap=%.2f | progress=%s | num_workers=%d"
        % (
            demucs_audio.shape[-1],
            sample_rate,
            target_sample_rate,
            device,
            demucs_overlap,
            demucs_progress,
            demucs_num_workers,
        )
    )
    started_at = time.perf_counter()
    with torch.inference_mode():
        estimates = apply_model(
            model,
            mixture,
            device=device,
            shifts=1,
            split=True,
            overlap=demucs_overlap,
            progress=demucs_progress,
            num_workers=demucs_num_workers,
        )
    _log("demucs done | elapsed=%.1fs" % (time.perf_counter() - started_at))

    if estimates.dim() == 4:
        estimates = estimates[0]

    sources = [str(source) for source in getattr(model, "sources", [])]
    if not sources:
        raise ValueError("Demucs model did not expose any source names")

    estimates = estimates.detach().cpu()
    source_estimates: dict[str, torch.Tensor] = {}
    for source_name, source_audio in zip(sources, estimates):
        restored_audio = source_audio
        if sample_rate != target_sample_rate:
            restored_audio = F.resample(restored_audio, target_sample_rate, sample_rate)
        source_estimates[source_name] = _restore_original_channels(
            restored_audio,
            original_channels,
        ).detach().cpu()
    return source_estimates


def separate(
    model,
    processor,
    device,
    audio: torch.Tensor,
    description: str,
    sample_rate: int | None = None,
    sr: int | None = None,
):
    effective_sample_rate = sample_rate if sample_rate is not None else sr
    if hasattr(model, "separate") and processor is not None:
        return _separate_with_sam(model, processor, device, audio, description, effective_sample_rate)

    if hasattr(model, "sources"):
        return _separate_with_demucs(model, device, audio, description, effective_sample_rate)

    raise TypeError(
        "Unsupported separation model. Expected a SAM Audio model or a Demucs model."
    )


def _resolve_settings(audio: torch.Tensor, sr: int | None) -> dict[str, int | bool | None]:
    chunk_seconds = float(os.environ.get("SAM_AUDIO_CHUNK_SECONDS", str(DEFAULT_CHUNK_SECONDS)))
    chunk_samples: int | None = None
    if chunk_seconds > 0:
        if sr is not None and sr > 0:
            chunk_samples = max(int(chunk_seconds * sr), 1)
        else:
            fallback = os.environ.get("SAM_AUDIO_CHUNK_SAMPLES")
            if fallback is not None:
                chunk_samples = max(int(fallback), 1)

    return {
        "chunk_samples": chunk_samples,
        "reranking_candidates": max(int(os.environ.get("SAM_AUDIO_RERANKING_CANDIDATES", str(DEFAULT_RERANKING_CANDIDATES))), 1),
        "oom_reranking_candidates": max(int(os.environ.get("SAM_AUDIO_OOM_RERANKING_CANDIDATES", str(DEFAULT_OOM_RERANKING_CANDIDATES))), 1),
        "predict_spans": _env_flag("SAM_AUDIO_PREDICT_SPANS", DEFAULT_PREDICT_SPANS),
    }


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _log(message: str) -> None:
    print(f"[sam-audio] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)
