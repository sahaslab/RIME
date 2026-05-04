from __future__ import annotations
from collections.abc import Callable, Iterable, Sequence
import numpy as np

EPSILON = 1e-8


def _prepare_audio(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    audio_array = np.asarray(audio, dtype=np.float32)
    if audio_array.ndim == 1:
        return audio_array[np.newaxis, :].copy(), True
    if audio_array.ndim == 2:
        return audio_array.copy(), False
    raise ValueError("audio must be a 1D or 2D numpy array")


def _restore_audio_shape(audio: np.ndarray, was_mono: bool) -> np.ndarray:
    if was_mono:
        return audio[0]
    return audio


def _db_to_amplitude(level_db: float) -> float:
    return float(10.0 ** (level_db / 20.0))


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + EPSILON))


def _peak_limit(audio: np.ndarray, ceiling: float = 0.99) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak <= ceiling:
        return audio.astype(np.float32, copy=False)
    return (audio * (ceiling / peak)).astype(np.float32, copy=False)


def _scale_artifact(
    artifact: np.ndarray,
    reference_audio: np.ndarray,
    level_db: float,
) -> np.ndarray:
    artifact_rms = _rms(artifact)
    if artifact_rms <= EPSILON:
        return artifact

    target_rms = max(_rms(reference_audio), EPSILON) * _db_to_amplitude(level_db)
    return artifact * (target_rms / artifact_rms)


def _fft_process(audio: np.ndarray, sr: int, response: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(audio, axis=-1)
    shaped = spectrum * response[np.newaxis, :]
    return np.fft.irfft(shaped, n=audio.shape[-1], axis=-1).astype(np.float32)


def _highpass_response(freqs: np.ndarray, cutoff_hz: float, order: int = 6) -> np.ndarray:
    safe_freqs = np.maximum(freqs, 1.0)
    return 1.0 / np.sqrt(1.0 + (cutoff_hz / safe_freqs) ** (2 * order))


def _lowpass_response(freqs: np.ndarray, cutoff_hz: float, order: int = 6) -> np.ndarray:
    safe_freqs = np.maximum(freqs, 1.0)
    return 1.0 / np.sqrt(1.0 + (safe_freqs / cutoff_hz) ** (2 * order))


def _bandpass_filter(
    audio: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    order: int = 6,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(audio.shape[-1], 1.0 / sr)
    response = _highpass_response(freqs, low_hz, order) * _lowpass_response(
        freqs, high_hz, order
    )
    return _fft_process(audio, sr, response.astype(np.float32))


def _highpass_filter(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    order: int = 6,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(audio.shape[-1], 1.0 / sr)
    response = _highpass_response(freqs, cutoff_hz, order)
    return _fft_process(audio, sr, response.astype(np.float32))


def _peak_eq(
    audio: np.ndarray,
    sr: int,
    center_frequency_hz: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(audio.shape[-1], 1.0 / sr)
    safe_freqs = np.maximum(freqs, 1.0)
    bandwidth_octaves = max(0.08, 1.0 / max(q, 0.2))
    distance = (
        np.log2(safe_freqs) - np.log2(max(center_frequency_hz, 1.0))
    ) / bandwidth_octaves
    bell = np.exp(-0.5 * np.square(distance))
    response = 1.0 + (_db_to_amplitude(gain_db) - 1.0) * bell
    return _fft_process(audio, sr, response.astype(np.float32))


def _high_shelf_eq(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float,
    gain_db: float,
    q: float,
) -> np.ndarray:
    freqs = np.fft.rfftfreq(audio.shape[-1], 1.0 / sr)
    safe_freqs = np.maximum(freqs, 1.0)
    steepness = max(2.0, 6.0 * q)
    transition = 1.0 / (
        1.0
        + np.exp(-(np.log2(safe_freqs) - np.log2(max(cutoff_hz, 1.0))) * steepness)
    )
    response = 1.0 + (_db_to_amplitude(gain_db) - 1.0) * transition
    return _fft_process(audio, sr, response.astype(np.float32))


def _make_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1:
        return signal
    kernel = np.ones(window_size, dtype=np.float32) / window_size
    return np.convolve(signal, kernel, mode="same")


def introduce_mains_hum(
    audio: np.ndarray,
    sr: int,
    fundamental_hz: float = 60.0,
    level_db: float = -28.0,
    harmonic_weights: Sequence[float] = (1.0, 0.45, 0.2),
    modulation_hz: float = 0.35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add low-level AC mains hum that can be removed with filtering/notching."""
    prepared_audio, was_mono = _prepare_audio(audio)
    channel_count, sample_count = prepared_audio.shape
    time = np.arange(sample_count, dtype=np.float32) / float(sr)
    local_rng = _make_rng(rng)

    modulation = 0.85 + 0.15 * np.sin(
        2.0 * np.pi * modulation_hz * time + local_rng.uniform(0.0, 2.0 * np.pi)
    )

    hum = np.zeros((1, sample_count), dtype=np.float32)
    for harmonic_index, weight in enumerate(harmonic_weights, start=1):
        phase = local_rng.uniform(0.0, 2.0 * np.pi)
        hum += weight * np.sin(
            2.0 * np.pi * fundamental_hz * harmonic_index * time + phase
        )

    hum *= modulation[np.newaxis, :]
    hum = np.repeat(hum, channel_count, axis=0)
    hum = _scale_artifact(hum, prepared_audio, level_db)
    return _restore_audio_shape(_peak_limit(prepared_audio + hum), was_mono)


def introduce_broadband_hiss(
    audio: np.ndarray,
    sr: int,
    level_db: float = -34.0,
    highpass_hz: float = 4500.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add high-frequency hiss similar to noisy preamps or rough headphone bleed."""
    prepared_audio, was_mono = _prepare_audio(audio)
    local_rng = _make_rng(rng)
    hiss = local_rng.normal(0.0, 1.0, size=prepared_audio.shape).astype(np.float32)
    hiss = _highpass_filter(hiss, sr, highpass_hz)
    hiss = _scale_artifact(hiss, prepared_audio, level_db)
    return _restore_audio_shape(_peak_limit(prepared_audio + hiss), was_mono)


def introduce_low_frequency_rumble(
    audio: np.ndarray,
    sr: int,
    level_db: float = -24.0,
    highpass_hz: float = 18.0,
    lowpass_hz: float = 45.0,
    modulation_hz: float = 0.25,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add sub-bass rumble from mic stands, HVAC, or floor vibration."""
    prepared_audio, was_mono = _prepare_audio(audio)
    local_rng = _make_rng(rng)
    rumble = local_rng.normal(0.0, 1.0, size=prepared_audio.shape).astype(np.float32)
    rumble = _bandpass_filter(rumble, sr, highpass_hz, lowpass_hz)

    time = np.arange(prepared_audio.shape[1], dtype=np.float32) / float(sr)
    modulation = 0.7 + 0.3 * np.sin(
        2.0 * np.pi * modulation_hz * time + local_rng.uniform(0.0, 2.0 * np.pi)
    )
    rumble *= modulation[np.newaxis, :]
    rumble = _scale_artifact(rumble, prepared_audio, level_db)
    return _restore_audio_shape(_peak_limit(prepared_audio + rumble), was_mono)


def introduce_sibilance(
    audio: np.ndarray,
    sr: int,
    boost_db: float = 10.0,
    band_start_hz: float = 4500.0,
    band_stop_hz: float = 10000.0,
    emphasis_quantile: float = 0.7,
    smoothing_ms: float = 8.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Exaggerate the sibilant band so a de-esser or high-shelf cut can tame it."""
    del rng
    prepared_audio, was_mono = _prepare_audio(audio)
    sibilant_band = _bandpass_filter(prepared_audio, sr, band_start_hz, band_stop_hz)

    band_env = np.mean(np.abs(sibilant_band), axis=0)
    full_env = np.mean(np.abs(prepared_audio), axis=0) + EPSILON
    brightness_ratio = band_env / full_env

    threshold = float(np.quantile(brightness_ratio, emphasis_quantile))
    activation = np.clip(
        (brightness_ratio - threshold) / (threshold + EPSILON),
        0.0,
        1.0,
    )
    smoothing_samples = max(1, int(sr * smoothing_ms / 1000.0))
    activation = _moving_average(activation.astype(np.float32), smoothing_samples)
    activation = activation[np.newaxis, :]

    boosted_band = sibilant_band * (_db_to_amplitude(boost_db) - 1.0) * activation
    return _restore_audio_shape(_peak_limit(prepared_audio + boosted_band), was_mono)


def introduce_boxy_resonance(
    audio: np.ndarray,
    sr: int,
    center_frequency_hz: float = 350.0,
    gain_db: float = 6.0,
    q: float = 1.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Boost muddy low-mids that are commonly cleaned up with parametric EQ."""
    del rng
    prepared_audio, was_mono = _prepare_audio(audio)
    processed = _peak_eq(prepared_audio, sr, center_frequency_hz, gain_db, q)
    return _restore_audio_shape(_peak_limit(processed), was_mono)


def introduce_harsh_presence(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float = 3200.0,
    gain_db: float = 5.0,
    q: float = 0.707,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Boost upper mids/highs to simulate harsh, under-EQ'd source material."""
    del rng
    prepared_audio, was_mono = _prepare_audio(audio)
    processed = _high_shelf_eq(prepared_audio, sr, cutoff_hz, gain_db, q)
    return _restore_audio_shape(_peak_limit(processed), was_mono)


ERROR_LIBRARY: dict[str, Callable[..., np.ndarray]] = {
    "mains_hum": introduce_mains_hum,
    "broadband_hiss": introduce_broadband_hiss,
    "low_frequency_rumble": introduce_low_frequency_rumble,
    "sibilance": introduce_sibilance,
    "boxy_resonance": introduce_boxy_resonance,
    "harsh_presence": introduce_harsh_presence,
}


def list_error_types() -> tuple[str, ...]:
    """Return the available degradations that can be sampled for data generation."""
    return tuple(ERROR_LIBRARY.keys())


def apply_error_chain(
    audio: np.ndarray,
    sr: int,
    error_names: Iterable[str],
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a sequence of named degradations to an audio array."""
    processed = np.asarray(audio, dtype=np.float32)
    local_rng = _make_rng(rng)

    for error_name in error_names:
        try:
            error_fn = ERROR_LIBRARY[error_name]
        except KeyError as exc:
            available = ", ".join(sorted(ERROR_LIBRARY))
            raise ValueError(
                f"Unknown error '{error_name}'. Available errors: {available}"
            ) from exc
        processed = error_fn(processed, sr, rng=local_rng)

    return processed
