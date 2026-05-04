import numpy as np
import torch
import psola
from skey.hcqt import VQT, CropCQT
from pedalboard import Pedalboard, PitchShift
from skey.chromanet import ChromaNet
from skey.key_detection import infer_key
from .pitch_utils import estimate_f0, quantize_to_key


def autotune(
    audio: torch.Tensor,
    sr: int,
    hcqt: VQT,
    chromanet: ChromaNet,
    crop_fn: CropCQT,
    device: str,
    key: str = None,
    mode: str = None,
) -> torch.Tensor:
    """
    Automatically quantize input vocals to a key and resnythesize
    """

    # Get initial f0 estimate
    audio_mono = torch.mean(audio, dim=0, keepdim=True).cpu()
    f0 = estimate_f0(audio_mono, sr).numpy()

    # Estimate key if it is not given
    if key is None or mode is None:
        key, mode = infer_key(hcqt, chromanet, crop_fn, audio_mono, device).split(" ")

    # Quantize input pitch values to target key
    target_f0 = quantize_to_key(f0, key, mode).flatten()

    # Resynthesize with psola
    audio_np = audio_mono.numpy().flatten()
    autotuned = psola.vocode(audio=audio_np, sample_rate=sr, target_pitch=target_f0)

    return torch.Tensor(autotuned)


def generate_harmony(
    audio: torch.Tensor,
    sr: int,
    hcqt: VQT,
    chromanet: ChromaNet,
    crop_fn: CropCQT,
    device: str,
    semitones: float,
    key: str = None,
    mode: str = None,
) -> torch.Tensor:
    """
    Generate a harmony with a distance of a specified number of semitones
    """

    # Get initial f0 estimate
    audio_mono = torch.mean(audio, dim=0, keepdim=True).cpu()
    f0 = estimate_f0(audio_mono, sr).numpy()

    # Estimate key if it is not given
    if key is None or mode is None:
        key, mode = infer_key(hcqt, chromanet, crop_fn, audio_mono, device).split(" ")

    # Quantize input pitch values to target key

    # Resynthesize with psola
    shift_factor = 2 ** (semitones / 12)
    rough_target_f0 = f0 * shift_factor
    target_f0 = quantize_to_key(rough_target_f0, key, mode).flatten()

    harmony = psola.vocode(
        audio=audio_mono.numpy().flatten(), sample_rate=sr, target_pitch=target_f0
    )

    return torch.Tensor(harmony)


def apply_pitch_shift(
    audio: np.ndarray,
    sr: int,
    semitones: float = 0.0,
) -> np.ndarray:
    """Apply a constant pitch shift to audio."""
    board = Pedalboard([PitchShift(semitones)])
    return board(audio, sr)
