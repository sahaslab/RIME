import numpy as np
import torch
import torchcrepe


def estimate_f0(
    audio: torch.Tensor,
    sr: int,
    model: str = "tiny",
    device: str = "cpu",
    fmin: float = 50,
    fmax: float = 550,
    batch_size: int = 2048,
) -> torch.Tensor:
    """
    See torchcrepe.predict for Args and Return structure
    """
    hop_length = int(sr / 200.0)
    pitch = torchcrepe.predict(
        audio, sr, hop_length, fmin, fmax, model, batch_size=batch_size, device=device
    )
    return pitch


def quantize_to_key(
    f0_array: np.ndarray,
    key_root: str = "C",
    scale_type: str = "major",
    ref_freq: float = 440.0,
) -> np.ndarray:
    """
    Quantizes f0 values to the nearest notes in a specific musical key.

    Args:
        f0_array (array-like): Input frequencies in Hz.
        key_root (str): The root note (e.g., 'C', 'F#', 'Eb').
        scale_type (str): 'major' or 'minor'.
        ref_freq (float): Reference frequency (A4).

    Returns:
        np.ndarray: Quantized frequencies in the specified key.
    """
    f0 = np.array(f0_array, dtype=float)
    out_f0 = np.zeros_like(f0)

    # 1. Define Scale Intervals (semitones from root)
    scales = {
        "major": [0, 2, 4, 5, 7, 9, 11],  # W-W-H-W-W-W-H
        "minor": [0, 2, 3, 5, 7, 8, 10],  # Natural Minor
        "chromatic": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }

    # Map note names to standard MIDI pitch classes (C=0, C#=1, etc.)
    note_map = {
        "C": 0,
        "C#": 1,
        "Db": 1,
        "D": 2,
        "D#": 3,
        "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "Gb": 6,
        "G": 7,
        "G#": 8,
        "Ab": 8,
        "A": 9,
        "A#": 10,
        "Bb": 10,
        "B": 11,
    }

    if key_root not in note_map:
        raise ValueError(f"Invalid key root: {key_root}")

    root_class = note_map[key_root]
    allowed_intervals = np.array(scales.get(scale_type, scales["major"]))

    # 2. Pre-calculate a 'Snap Map' for efficiency
    # This creates a lookup table of size 12. For every possible chromatic note (0-11),
    # it stores the offset to the nearest valid scale note.
    snap_map = np.zeros(12)
    for i in range(12):
        # Calculate distance to all allowed intervals (handling wrap-around for distance)
        # We calculate distances in both directions and take the minimum
        dist_direct = np.abs(allowed_intervals - i)
        dist_wrap = 12 - dist_direct
        min_dist = np.minimum(dist_direct, dist_wrap)

        # Find the index of the closest allowed interval
        closest_idx = np.argmin(min_dist)
        target_interval = allowed_intervals[closest_idx]

        # Calculate the shift needed (e.g., +1 or -1 semitone)
        # We use standard modular arithmetic distance
        diff = target_interval - i
        # Adjust for shortest path (e.g., if i=11 and target=0, shift is +1, not -11)
        if diff > 6:
            diff -= 12
        elif diff < -6:
            diff += 12
        snap_map[i] = diff

    # 3. Process the Audio Data
    nonzero_mask = f0 > 0
    if np.any(nonzero_mask):
        # Convert Hz to continuous MIDI note number (A4 = 69)
        midi_vals = 69 + 12 * np.log2(f0[nonzero_mask] / ref_freq)

        # Round to nearest integer (chromatic pitch)
        midi_rounded = np.round(midi_vals)

        # Determine the pitch class (0-11) relative to the chosen key root
        # (midi_rounded - root_class) % 12 gives us the interval from the root
        current_intervals = (midi_rounded.astype(int) - root_class) % 12

        # Look up the correction needed using our snap_map
        corrections = snap_map[current_intervals]

        # Apply correction to get the snapped MIDI note
        snapped_midi = midi_rounded + corrections

        # Convert back to Hz
        out_f0[nonzero_mask] = ref_freq * (2 ** ((snapped_midi - 69) / 12))

    return out_f0
