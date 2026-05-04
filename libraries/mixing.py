import numpy as np
from pedalboard import Gain, NoiseGate, PeakFilter, Pedalboard, LowpassFilter, HighpassFilter, LowShelfFilter, HighShelfFilter


def apply_gain(audio: np.ndarray, sr: int, gain_db: float = 1.0) -> np.ndarray:
    """Apply gain to audio."""
    board = Pedalboard([Gain(gain_db)])
    return board(audio, sr)


def apply_highpass(
    audio: np.ndarray, sr: int, cutoff_frequency_hz: float = 50.0
) -> np.ndarray:
    """Apply highpass filter to audio."""
    board = Pedalboard([HighpassFilter(cutoff_frequency_hz)])
    return board(audio, sr)


def apply_lowpass(
    audio: np.ndarray, sr: int, cutoff_frequency_hz: float = 50.0
) -> np.ndarray:
    """Apply lowpass filter to audio."""
    board = Pedalboard([LowpassFilter(cutoff_frequency_hz)])
    return board(audio, sr)


def apply_highshelf(
    audio: np.ndarray,
    sr: int,
    cutoff_frequency_hz: float = 440.0,
    gain_db: float = 0.0,
    q: float = 0.7071067690849304,
) -> np.ndarray:
    """Apply high shelf filter to audio."""
    board = Pedalboard([HighShelfFilter(cutoff_frequency_hz, gain_db, q)])
    return board(audio, sr)


def apply_lowshelf(
    audio: np.ndarray,
    sr: int,
    cutoff_frequency_hz: float = 440.0,
    gain_db: float = 0.0,
    q: float = 0.7071067690849304,
) -> np.ndarray:
    """Apply low shelf filter to audio."""
    board = Pedalboard([LowShelfFilter(cutoff_frequency_hz, gain_db, q)])
    return board(audio, sr)


def apply_peakfilter(
    audio: np.ndarray,
    sr: int,
    cutoff_frequency_hz: float = 440.0,
    gain_db: float = 0.0,
    q: float = 0.7071067690849304,
) -> np.ndarray:
    """Apply peak filter (parametric EQ) to audio."""
    board = Pedalboard([PeakFilter(cutoff_frequency_hz, gain_db, q)])
    return board(audio, sr)


def apply_noisegate(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -100.0,
    ratio: float = 10.0,
    attack_ms: float = 1.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    """Apply noise gate to audio."""
    board = Pedalboard([NoiseGate(threshold_db, ratio, attack_ms, release_ms)])
    return board(audio, sr)


def normalize_peak(
    audio: np.ndarray,
    sr: int,
    target_peak: float = 0.95,
) -> np.ndarray:
    """Scale audio so its maximum absolute sample reaches target_peak."""
    if not 0.0 < target_peak <= 1.0:
        raise ValueError("target_peak must be within (0.0, 1.0]")

    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio.copy()

    return audio * (target_peak / peak)


def apply_fade_in_out(
    audio: np.ndarray,
    sr: int,
    fade_in_seconds: float = 0.01,
    fade_out_seconds: float = 0.01,
) -> np.ndarray:
    """Apply linear fade in and fade out to prevent clicks at edit boundaries."""
    if fade_in_seconds < 0.0 or fade_out_seconds < 0.0:
        raise ValueError("fade durations must be non-negative")

    num_samples = audio.shape[-1]
    fade_in_samples = min(int(round(fade_in_seconds * sr)), num_samples)
    fade_out_samples = min(int(round(fade_out_seconds * sr)), num_samples)

    envelope = np.ones(num_samples, dtype=audio.dtype)
    if fade_in_samples > 0:
        envelope[:fade_in_samples] = np.linspace(
            0.0, 1.0, fade_in_samples, endpoint=True, dtype=audio.dtype
        )
    if fade_out_samples > 0:
        envelope[-fade_out_samples:] = np.linspace(
            1.0, 0.0, fade_out_samples, endpoint=True, dtype=audio.dtype
        )

    return audio * envelope


def apply_pan(
    audio: np.ndarray,
    sr: int,
    pan: float = 0.0,
) -> np.ndarray:
    """Apply constant-power stereo panning. Negative is left, positive is right."""
    if not -1.0 <= pan <= 1.0:
        raise ValueError("pan must be within [-1.0, 1.0]")

    if audio.ndim != 2:
        raise ValueError("audio must have shape [channels, frames]")
    if audio.shape[0] != 2:
        raise ValueError("apply_pan requires stereo audio with exactly 2 channels")

    angle = (pan + 1.0) * (np.pi / 4.0)
    left_gain = np.cos(angle)
    right_gain = np.sin(angle)

    panned = audio.copy()
    panned[0] *= left_gain
    panned[1] *= right_gain
    return panned


def mix_stem_with_residual(stem: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Mix a stem with its residual."""
    return stem + residual
