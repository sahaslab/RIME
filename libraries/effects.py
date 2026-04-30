import numpy as np
from scipy import signal
from pedalboard import Delay, Chorus, Phaser, Reverb, Limiter, Compressor, Distortion, Pedalboard


def _compute_rms(audio: np.ndarray) -> float:
    """Compute RMS (Root Mean Square) of an audio signal."""
    return np.sqrt(np.mean(audio ** 2))


def _compute_crest_factor_db(audio: np.ndarray) -> float:
    """
    Crest factor = peak / RMS, expressed in dB.
    Higher values indicate more transient/percussive content.
      ~10–15 dB → sustained sources (vocals, pads)
      ~18–25 dB → percussive sources (drums, plucked strings)
    """
    rms = _compute_rms(audio)
    peak = np.max(np.abs(audio))
    if rms < 1e-10:
        return 0.0
    return 20 * np.log10(peak / rms)

def _gain_from_db(db: float) -> float:
    return 10 ** (db / 20.0)

def _crest_factor_blend(
    audio_pre: np.ndarray,
    audio_post: np.ndarray,
    crest_db: float,
    rms_threshold_db: float = 12.0,   # below → pure RMS
    peak_threshold_db: float = 20.0,  # above → pure peak
) -> float:
    """
    Blend RMS and peak makeup gain based on crest factor.
    Returns a linear gain multiplier.

    Blending curve:
        crest < rms_threshold  → weight = 0.0  (full RMS correction)
        crest > peak_threshold → weight = 1.0  (full peak correction)
        in between             → linear interpolation
    """
    rms_gain  = _compute_rms(audio_pre)  / max(_compute_rms(audio_post),  1e-10)
    peak_gain = np.max(np.abs(audio_pre)) / max(np.max(np.abs(audio_post)), 1e-10)

    # Normalise crest factor into [0, 1] blend weight
    weight = np.clip(
        (crest_db - rms_threshold_db) / (peak_threshold_db - rms_threshold_db),
        0.0, 1.0
    )

    return (1.0 - weight) * rms_gain + weight * peak_gain

def apply_chorus(
    audio: np.ndarray,
    sr: int,
    rate_hz: float = 1.0,
    depth: float = 0.25,
    centre_delay_ms: float = 7.0,
    feedback: float = 0.0,
    mix: float = 0.5,
) -> np.ndarray:
    """Apply chorus effect to audio."""
    board = Pedalboard([Chorus(rate_hz, depth, centre_delay_ms, feedback, mix)])
    return board(audio, sr)


def apply_phaser(
    audio: np.ndarray,
    sr: int,
    rate_hz: float = 1.0,
    depth: float = 0.5,
    centre_frequency_hz: float = 1300.0,
    feedback: float = 0.0,
    mix: float = 0.5,
) -> np.ndarray:
    """Apply phaser effect to audio."""
    board = Pedalboard([Phaser(rate_hz, depth, centre_frequency_hz, feedback, mix)])
    return board(audio, sr)


def apply_distortion(
    audio: np.ndarray,
    sr: int,
    drive_db: float = 25.0,
) -> np.ndarray:
    """Apply distortion effect to audio."""
    board = Pedalboard([Distortion(drive_db)])
    return board(audio, sr)


def apply_reverb(
    audio: np.ndarray,
    sr: int,
    room_size: float = 0.5,
    damping: float = 0.5,
    wet_level: float = 0.33,
    dry_level: float = 0.4,
    width: float = 1.0,
    freeze_mode: float = 0.0,
) -> np.ndarray:
    """Apply reverb effect to audio."""
    board = Pedalboard(
        [Reverb(room_size, damping, wet_level, dry_level, width, freeze_mode)]
    )
    return board(audio, sr)


def apply_delay(
    audio: np.ndarray,
    sr: int,
    delay_seconds: float = 0.5,
    feedback: float = 0.0,
    mix: float = 0.5,
) -> np.ndarray:
    """Apply delay effect to audio."""
    board = Pedalboard([Delay(delay_seconds, feedback, mix)])
    return board(audio, sr)


def apply_compressor(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = 0.0,
    ratio: float = 1.0,
    attack_ms: float = 1.0,
    release_ms: float = 10.0,
) -> np.ndarray:
    """Apply compressor to audio."""
    board = Pedalboard([Compressor(threshold_db, ratio, attack_ms, release_ms)])
    compressed = board(audio, sr)

    # Gain dynamics correction
    crest_db = _compute_crest_factor_db(audio)
    makeup_linear = _crest_factor_blend(audio, compressed, crest_db)
    corrected = compressed * makeup_linear
    ceiling = _gain_from_db(-1.0)
    peak = np.max(np.abs(corrected))
    if peak > ceiling:
        corrected *= ceiling / peak
    return corrected


def apply_limiter(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -10.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    """Apply limiter to audio."""
    board = Pedalboard([Limiter(threshold_db, release_ms)])
    return board(audio, sr)


def _to_channel_first(audio: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Convert audio to (channels, samples).
    Supports:
      - (samples,)
      - (channels, samples)
      - (samples, channels)
    """
    x = np.asarray(audio, dtype=np.float32)

    if x.ndim == 1:
        return x[None, :], "mono"

    if x.ndim != 2:
        raise ValueError("audio must be 1D or 2D")

    # Heuristic: if first dim is small, treat as channels-first
    if x.shape[0] <= 8:
        return x, "channels_first"

    return x.T, "samples_first"


def _from_channel_first(x: np.ndarray, layout: str) -> np.ndarray:
    if layout == "mono":
        return x[0]
    if layout == "channels_first":
        return x
    if layout == "samples_first":
        return x.T
    raise ValueError(f"Unknown layout: {layout}")


def _lin_to_db(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, eps))


def _envelope_follower(
    x: np.ndarray,
    sr: int,
    attack_ms: float,
    release_ms: float,
) -> np.ndarray:
    """
    Attack/release peak envelope follower.
    x must be non-negative, shape (channels, samples).
    """
    attack_s = max(attack_ms, 0.01) * 1e-3
    release_s = max(release_ms, 0.01) * 1e-3

    attack_coeff = np.exp(-1.0 / (sr * attack_s))
    release_coeff = np.exp(-1.0 / (sr * release_s))

    env = np.zeros_like(x, dtype=np.float32)

    for ch in range(x.shape[0]):
        prev = 0.0
        for n in range(x.shape[1]):
            target = x[ch, n]
            coeff = attack_coeff if target > prev else release_coeff
            prev = coeff * prev + (1.0 - coeff) * target
            env[ch, n] = prev

    return env


def _bandpass_filter(
    x_cf: np.ndarray,
    sr: int,
    low_hz: float,
    high_hz: float,
    order: int = 6,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass on channel-first audio.
    """
    if low_hz >= high_hz:
        raise ValueError("low_hz must be lower than high_hz")
    if low_hz <= 0:
        raise ValueError("low_hz must be > 0")
    if high_hz >= sr / 2:
        raise ValueError("high_hz must be below Nyquist")

    sos = signal.butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sr,
        output="sos",
    )

    # Avoid padlen issues on very short clips
    if x_cf.shape[1] < 64:
        return signal.sosfiltfilt(sos, x_cf, axis=1, padlen=0)

    return signal.sosfiltfilt(sos, x_cf, axis=1)


def apply_deesser(
    audio: np.ndarray,
    sr: int,
    ess_low_hz: float = 5500.0,
    ess_high_hz: float = 8500.0,
    threshold_db: float = -34.0,
    ratio: float = 6.0,
    attack_ms: float = 0.5,
    release_ms: float = 80.0,
    relative_threshold_db: float = -12.0,
    max_reduction_db: float = 12.0,
    filter_order: int = 6,
) -> np.ndarray:
    """
    Split-band de-esser
    """
    if ess_low_hz >= ess_high_hz:
        raise ValueError("ess_low_hz must be lower than ess_high_hz")
    if ratio < 1.0:
        raise ValueError("ratio must be >= 1.0")

    x_cf, layout = _to_channel_first(audio)

    # 1) Isolate sibilance band more cleanly
    ess_band = _bandpass_filter(
        x_cf,
        sr,
        low_hz=ess_low_hz,
        high_hz=ess_high_hz,
        order=filter_order,
    )

    # 2) Envelope detection
    ess_env = _envelope_follower(np.abs(ess_band), sr, attack_ms, release_ms)
    full_env = _envelope_follower(np.abs(x_cf), sr, attack_ms, release_ms)

    ess_db = _lin_to_db(ess_env)
    rel_db = _lin_to_db(ess_env / np.maximum(full_env, 1e-12))

    # 3) Compression law:
    #    only reduce when:
    #    - ess band is above threshold_db
    #    - ess band is relatively dominant vs full vocal
    over_db = np.maximum(ess_db - threshold_db, 0.0)
    active = rel_db > relative_threshold_db

    gain_reduction_db = np.where(
        active,
        -(1.0 - 1.0 / ratio) * over_db,
        0.0,
    )

    gain_reduction_db = np.maximum(gain_reduction_db, -max_reduction_db)
    gain = 10.0 ** (gain_reduction_db / 20.0)

    # 4) Apply gain reduction only to the ess band
    ess_band_deessed = ess_band * gain

    # 5) Recombine
    remainder = x_cf - ess_band
    out = remainder + ess_band_deessed

    return np.clip(_from_channel_first(out, layout), -1.0, 1.0)
