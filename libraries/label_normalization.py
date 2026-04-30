from __future__ import annotations
import re
from dataclasses import dataclass
from collections.abc import Sequence

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class NormalizationChoice:
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizationResult:
    query: str
    normalized_query: str
    canonical: str | None
    confidence: float
    source: str
    rationale: str | None = None


def normalize_label(
    query: str,
    choices: Sequence[NormalizationChoice | str],
) -> NormalizationResult:
    resolved_choices = tuple(_coerce_choice(choice) for choice in choices)
    normalized_query = _normalize_text(query)

    direct = _direct_match(normalized_query, resolved_choices)
    if direct is not None:
        return NormalizationResult(
            query=query,
            normalized_query=normalized_query,
            canonical=direct,
            confidence=1.0,
            source="alias",
        )

    return NormalizationResult(
        query=query,
        normalized_query=normalized_query,
        canonical=None,
        confidence=0.0,
        source="unmatched",
    )


def _coerce_choice(choice: NormalizationChoice | str) -> NormalizationChoice:
    if isinstance(choice, NormalizationChoice):
        return choice
    return NormalizationChoice(name=str(choice))


def _normalize_text(value: str) -> str:
    return _NON_ALNUM_RE.sub(" ", value.strip().lower()).strip()


def _direct_match(normalized_query: str, choices: Sequence[NormalizationChoice]) -> str | None:
    if not normalized_query:
        return None

    alias_to_choice: dict[str, str] = {}
    for choice in choices:
        for alias in (choice.name, *choice.aliases):
            normalized_alias = _normalize_text(alias)
            if normalized_alias:
                alias_to_choice[normalized_alias] = choice.name

    if normalized_query in alias_to_choice:
        return alias_to_choice[normalized_query]

    query_tokens = set(normalized_query.split())
    for alias, choice_name in alias_to_choice.items():
        alias_tokens = set(alias.split())
        if alias_tokens and alias_tokens <= query_tokens:
            return choice_name
    return None
