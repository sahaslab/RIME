from __future__ import annotations
from collections.abc import Iterable
from libraries.label_normalization import NormalizationChoice, NormalizationResult, normalize_label

DEMUCS_6S_CHOICES: tuple[NormalizationChoice, ...] = (
    NormalizationChoice("vocals", ("vocal", "lead vocal", "backing vocals", "choir")),
    NormalizationChoice("drums", ("drum", "percussion", "beat", "kick", "snare")),
    NormalizationChoice("bass", ("sub", "low end", "bass guitar")),
    NormalizationChoice("guitar", ("electric guitar", "acoustic guitar", "elec gtr", "guitars")),
    NormalizationChoice("piano", ("keys", "keyboard", "grand piano")),
    NormalizationChoice("other", ("instrumental", "accompaniment", "music", "orchestra", "strings", "synth")),
)


def normalize_demucs_6s_target(query: str) -> NormalizationResult:
    return normalize_label(query, DEMUCS_6S_CHOICES)


def supported_demucs_6s_sources() -> tuple[str, ...]:
    return tuple(choice.name for choice in DEMUCS_6S_CHOICES)


def filter_target_candidates(
    candidates: Iterable[dict],
    profile: str | None,
) -> list[dict]:
    if profile in (None, "", "none"):
        return [dict(candidate) for candidate in candidates]
    if profile == "demucs_6s":
        return filter_demucs_6s_candidates(candidates)
    raise ValueError(f"Unsupported separation profile '{profile}'.")


def filter_demucs_6s_candidates(candidates: Iterable[dict]) -> list[dict]:
    filtered: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        stem = str(candidate.get("stem", ""))
        normalized = normalize_demucs_6s_target(stem)
        if normalized.canonical is None:
            continue
        updated = dict(candidate)
        updated["separation_target"] = normalized.canonical
        updated["separation_target_source"] = normalized.source
        key = (updated["separation_target"], str(updated.get("family", "other")))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(updated)
    return filtered
