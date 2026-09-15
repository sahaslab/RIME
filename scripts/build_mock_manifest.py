"""Generate mock analysis manifests by sampling stem combinations from priors.

Lets the planner be exercised without the real corpus: no audio, no dataset
metadata tables, no tagging pass. The planner only requires
`analysis.target_candidates`, so a mock manifest is essentially a sampled set of
Demucs stems per clip, drawn from configs/ground_truth/stem_priors.yaml.

Rows are built by `build_manifest_row`, the same factory the real dataset
adapters use, so a mock row is shape-identical to a real one -- including the
`priority` and `separation_target_source` fields on each target candidate. This
script picks instrument tags and hands them over; it never assembles
`target_candidates` itself, so the two paths cannot drift apart.

Everything except the stem combination is sampled from rates measured on the
same MusicCaps corpus the priors come from, so the defaults produce a corpus
that looks like the real one. Raise --issue-rate and --tempo-rate above those
defaults when you care more about reaching gated recipes than about realism;
see their help text.
"""

import sys
import json
import math
import random
import argparse
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_truth.datasets import build_manifest_row, load_dataset_config
from ground_truth.io_utils import write_jsonl

# Rates measured over the 5,200-clip MusicCaps analysis manifest. Only the
# genres the planner actually gates on are listed; everything else is invisible
# to it, so sampling it would be noise in the output for no behavioural gain.
# See configs/ground_truth/recipes.yaml for the gates.
GENRE_RATES = {
    "pop": 0.117,
    "rock": 0.104,
    "folk": 0.054,
    "classical": 0.052,
    "electronic": 0.047,
    "ambient": 0.040,
    "lounge": 0.001,
}

# Likewise for issues, which gate the corrective recipes. `sibilance` is defined
# in the MusicCaps dataset config but never actually occurs in the corpus, so it
# carries no measured rate and is omitted rather than invented.
ISSUE_RATES = {"hiss": 0.016, "hum": 0.004}

# MusicCaps ships no tempo at all, hence 0.0. Raising it is the only way to
# exercise the tempo-synced delay path; below that threshold the planner falls
# back to the 120 BPM prior in configs/ground_truth/motifs.yaml.
DEFAULT_TEMPO_RATE = 0.0
TEMPO_RANGE = (70.0, 160.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entries", type=int, default=100, help="Number of mock clips to generate. Default: %(default)s")
    parser.add_argument("--seed", type=int, default=0, help="Random seed, so a given corpus is reproducible. Default: %(default)s")
    parser.add_argument("--priors-path", type=Path, default=Path("configs/ground_truth/stem_priors.yaml"), help="Stem combination priors. Default: %(default)s")
    parser.add_argument("--profile", default=None, help="Priors profile to sample from. Default: the file's own default_profile")
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/ground_truth/datasets/musiccaps.yaml"), help="Dataset config supplying the instrument vocabulary and family maps. Default: %(default)s")
    parser.add_argument("--audio-root", type=Path, default=None, help="Root for synthetic audio paths. Default: paths are emitted but point nowhere, which is fine unless you intend to render")
    parser.add_argument("--clip-prefix", default="mock", help="Prefix for generated clip ids. Default: %(default)s")
    parser.add_argument("--issue-rate", type=float, default=None, help="Per-clip probability of carrying a recording issue. Default: the measured MusicCaps rate of %.3f, which is too low to reach the cleanup recipes in a small sample -- raise it to about 0.3 for coverage runs" % sum(ISSUE_RATES.values()))
    parser.add_argument("--tempo-rate", type=float, default=DEFAULT_TEMPO_RATE, help="Per-clip probability of carrying a tempo. Default: %(default)s, matching MusicCaps, which ships none. Raise it to exercise tempo-synced delay")
    parser.add_argument("--output-path", type=Path, default=Path("derived/ground_truth/mock_analysis_manifest.jsonl"), help="Output JSONL path. Default: %(default)s")
    args = parser.parse_args()

    if args.entries < 1:
        parser.error("--entries must be at least 1.")

    dataset_config = load_dataset_config(args.dataset_config)
    combinations, weights, profile_name = _load_priors(args.priors_path, args.profile)
    tag_choices = _stems_to_tags(dataset_config, combinations)

    rng = random.Random(args.seed)
    rows = [
        _build_row(
            index=index,
            rng=rng,
            combinations=combinations,
            weights=weights,
            tag_choices=tag_choices,
            dataset_config=dataset_config,
            audio_root=args.audio_root,
            clip_prefix=args.clip_prefix,
            issue_rate=args.issue_rate,
            tempo_rate=args.tempo_rate,
            profile_name=profile_name,
        )
        for index in range(args.entries)
    ]
    write_jsonl(args.output_path, rows)
    _print_summary(rows, args.output_path, profile_name, combinations, weights)


def _load_priors(priors_path: Path, profile: str | None) -> tuple[list[list[str]], list[float], str]:
    with priors_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    priors = loaded.get("stem_priors")
    if not priors:
        raise ValueError("Priors file '%s' is missing a 'stem_priors' root key." % priors_path)

    profiles = priors.get("profiles") or {}
    name = profile or priors.get("default_profile")
    if name not in profiles:
        raise ValueError(
            "Profile '%s' not found in '%s'. Available: %s"
            % (name, priors_path, ", ".join(sorted(profiles)))
        )

    values = (profiles[name].get("combinations") or {}).get("values") or []
    if not values:
        raise ValueError("Profile '%s' in '%s' has no combinations." % (name, priors_path))

    combinations = [list(entry["value"]) for entry in values]
    # Weights are relative, exactly as `random.choices` and the planner's own
    # choice distributions treat them, so they need no normalization here.
    weights = [float(entry.get("weight", 1.0)) for entry in values]
    return combinations, weights, name


def _stems_to_tags(
    dataset_config: Mapping[str, Any],
    combinations: Sequence[Sequence[str]],
) -> dict[str, list[str]]:
    """Invert `demucs_target_map` so a sampled stem can be expressed as a tag.

    `build_manifest_row` takes instrument tags and derives target candidates from
    them, which is the direction the real pipeline runs. Going stem -> tag here
    means the mock path rejoins that pipeline at the same point a real dataset
    adapter does, rather than fabricating candidates that could drift from what
    `_build_target_candidates` would actually produce.
    """
    demucs_target_map = dataset_config.get("demucs_target_map") or {}
    stem_map = dataset_config.get("target_stems") or {}

    tags: dict[str, list[str]] = {}
    for tag, separation_target in demucs_target_map.items():
        stem = stem_map.get(separation_target, separation_target)
        tags.setdefault(stem, []).append(tag)
    for stem in tags:
        tags[stem].sort()

    required = {stem for combination in combinations for stem in combination}
    missing = sorted(required - set(tags))
    if missing:
        raise ValueError(
            "The priors sample stems with no tag in the dataset config's "
            "demucs_target_map, so they cannot be expressed as a manifest row: %s"
            % ", ".join(missing)
        )
    return tags


def _build_row(
    index: int,
    rng: random.Random,
    combinations: Sequence[Sequence[str]],
    weights: Sequence[float],
    tag_choices: Mapping[str, list[str]],
    dataset_config: Mapping[str, Any],
    audio_root: Path | None,
    clip_prefix: str,
    issue_rate: float | None,
    tempo_rate: float,
    profile_name: str,
) -> dict[str, Any]:
    combination = rng.choices(list(combinations), weights=list(weights), k=1)[0]
    # One tag per stem. Real manifests often carry several tags resolving to the
    # same stem, but `_build_target_candidates` dedupes on (stem, family), so the
    # extras would not change the candidate list the planner sees.
    instrument_tags = [rng.choice(tag_choices[stem]) for stem in sorted(combination)]

    clip_id = "%s_%05d" % (clip_prefix, index)
    relative_path = "%s.wav" % clip_id
    audio_path = None if audio_root is None else str(audio_root / relative_path)

    genres = [genre for genre, rate in GENRE_RATES.items() if rng.random() < rate]
    issues = _sample_issues(rng, issue_rate)
    tempo_bpm = round(rng.uniform(*TEMPO_RANGE), 1) if rng.random() < tempo_rate else None

    return build_manifest_row(
        clip_id=clip_id,
        audio_path=audio_path,
        dataset_payload={
            "name": "mock",
            "track_id": clip_id,
            "track_name": clip_id,
            "relative_path": relative_path,
            # Recorded so a mock manifest is never mistaken for real data, and
            # so a stale one can be traced back to the priors that produced it.
            "mock_profile": profile_name,
        },
        genres=genres,
        mood_themes=(),
        instrument_tags=instrument_tags,
        dataset_config=dataset_config,
        tempo_bpm=tempo_bpm,
        issues=issues,
        caption=None,
    )


def _sample_issues(rng: random.Random, issue_rate: float | None) -> list[str]:
    """At most one issue per clip, matching how MusicCaps rows actually look.

    With `issue_rate` given, the measured per-issue rates become relative
    weights, so raising the rate for a coverage run keeps hiss more common than
    hum rather than flattening them.
    """
    if issue_rate is None:
        return [issue for issue, rate in ISSUE_RATES.items() if rng.random() < rate][:1]
    if rng.random() >= issue_rate:
        return []
    issues = sorted(ISSUE_RATES)
    return [rng.choices(issues, weights=[ISSUE_RATES[issue] for issue in issues], k=1)[0]]


def _print_summary(
    rows: Sequence[Mapping[str, Any]],
    output_path: Path,
    profile_name: str,
    prior_combinations: Sequence[Sequence[str]],
    prior_weights: Sequence[float],
) -> None:
    from collections import Counter

    total = len(rows)
    combinations: Counter[tuple[str, ...]] = Counter()
    stems: Counter[str] = Counter()
    with_issue = 0
    with_tempo = 0
    for row in rows:
        analysis = row["analysis"]
        present = tuple(sorted({c["stem"] for c in analysis["target_candidates"]}))
        combinations[present] += 1
        stems.update(present)
        with_issue += bool(analysis["issues"])
        with_tempo += analysis["tempo_bpm"] is not None

    # Show what the profile asks for beside what came out. The profile weights
    # are conditional on a non-empty combination, whereas the `diagnostics`
    # block in the priors file is unconditional, so reading the sampled rates
    # against that block makes a correct run look wrong by the empty-combination
    # mass. Printing the right baseline here removes the trap.
    scale = sum(prior_weights)
    expected = {
        stem: sum(
            weight
            for combination, weight in zip(prior_combinations, prior_weights)
            if stem in combination
        )
        / scale
        for stem in stems
    }
    error = math.sqrt(1.0 / total) if total else 0.0

    print("Wrote %d mock clips to %s (profile: %s)" % (total, output_path, profile_name))
    print("  stem presence (sampled vs. profile; +-%.3f is one sd at n=%d):" % (0.5 * error, total))
    for stem, count in stems.most_common():
        sampled = count / total
        target = expected[stem]
        deviations = abs(sampled - target) / math.sqrt(max(target * (1.0 - target) / total, 1e-12))
        print("    %-8s %.3f   profile %.3f   %+.1f sd" % (stem, sampled, target, deviations if sampled >= target else -deviations))
    print("  distinct combinations: %d" % len(combinations))
    print("  most common:")
    for combination, count in combinations.most_common(5):
        print("    %-38s %d" % (list(combination), count))
    print("  clips with an issue: %d   with a tempo: %d" % (with_issue, with_tempo))


if __name__ == "__main__":
    main()
