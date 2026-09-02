"""
Resolve free-text instrument names onto a dataset's instrument tag vocabulary.

Both the caption-derived adapters and the model-based tagger receive prose
("electric guitar", "shimmering hi hats", "soft female vocal") that has to land
on keys of a dataset config's `demucs_target_map`, since anything outside that
map is rejected for ground-truth planning.
"""

from __future__ import annotations
import re
from collections.abc import Mapping, Iterable, Sequence

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")

# Phrases like "no voices" or "without percussion" name an instrument that is
# absent. Resolving them would invent a separation target the clip cannot serve.
NEGATION_PREFIXES = ("no", "not", "without", "lacks", "lacking", "absent", "none", "sans", "minus")


def normalize_instrument_tags(
    raw_tags: Iterable[str],
    demucs_target_map: Mapping[str, str],
) -> tuple[list[str], list[str]]:
    """
    Map free-text instrument names onto the dataset config's tag vocabulary.

    Only keys of `demucs_target_map` survive; anything else is rejected for
    ground-truth planning. Returns (matched tags, unmatched raw tags), both
    deduplicated in first-seen order.
    """
    matched: list[str] = []
    unmatched: list[str] = []
    for raw_tag in raw_tags:
        resolved = resolve_instrument_tag(raw_tag, demucs_target_map)
        if resolved is None:
            unmatched.append(raw_tag)
        else:
            matched.append(resolved)
    return dedupe(matched), dedupe(unmatched)


def resolve_instrument_tag(raw_tag: str, demucs_target_map: Mapping[str, str]) -> str | None:
    """
    Resolve one free-text name to a vocabulary tag, most specific match first.

    The vocabulary concatenates words ("electricguitar"), so the whole phrase is
    squashed and looked up before falling back to the head noun: "electric
    guitar" -> electricguitar, "drum kit" -> drum, "lead vocals" -> vocals.
    Negated phrases resolve to nothing.
    """
    tokens = tokenize(raw_tag)
    if not tokens or is_negated(tokens):
        return None

    # Whole phrase, then each token from the head noun backwards.
    for candidate in ["".join(tokens), *reversed(tokens)]:
        for variant in _plural_variants(candidate):
            if variant in demucs_target_map:
                return variant
    return None


def tokenize(value: str) -> list[str]:
    """Lowercase alphanumeric tokens, punctuation and case discarded."""
    return _NON_ALNUM_RE.sub(" ", value.strip().lower()).split()


def is_negated(tokens: Sequence[str]) -> bool:
    """True when the phrase leads with a negation, as in "no percussion"."""
    return bool(tokens) and tokens[0] in NEGATION_PREFIXES


def matches_any_word(value: str, words: Iterable[str]) -> list[str]:
    """
    Which of `words` appear as whole tokens in `value`, in the order given.

    Used to mine genre and mood vocabularies out of free-text caption aspects,
    where "classic rock" should yield rock but "popular" should not yield pop.
    """
    tokens = set(tokenize(value))
    return [word for word in words if set(tokenize(word)) <= tokens and tokens]


def _plural_variants(value: str) -> list[str]:
    """The token itself plus its naive singular/plural forms, in match order."""
    if value.endswith("s"):
        return [value, value[:-1]]
    return [value, value + "s"]


def dedupe(items: Iterable[str]) -> list[str]:
    """Remove duplicates while preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
