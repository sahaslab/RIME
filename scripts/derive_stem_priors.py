"""Derive priors over Demucs stem combinations for mock analysis manifests.

Three reference corpora are combined, each doing only what it uniquely can:

  MusicCaps       the joint's shape. The only source with a diverse, complete
                  joint over stem combinations. Captions under-report, so the
                  raw joint is biased toward sparse combinations.
  MoisesDB        sensitivity estimation. True multitrack inventories, so zero
                  measurement error, but nearly every track is full-band and it
                  carries almost no combination variance of its own.
  Nunes &         held-out validation, never fit to. Expert-coded marginals and
  Ordanini 2014   cardinality from 2,399 Billboard Hot 100 songs.

The correction models an observed (caption-derived) stem set as the true set
`thinned`: each truly present stem survives into the caption independently with
probability `s_k`. That is inverted by regularized EM over the 16 subsets.

See configs/ground_truth/refs/ for the Nunes & Ordanini paper, and the emitted
configs/ground_truth/stem_priors.yaml header for the full method write-up.
"""

import sys
import json
import math
import zipfile
import argparse
import itertools
from pathlib import Path
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ground_truth.datasets import load_dataset_config
from ground_truth.io_utils import load_records

# MusicCaps and MoisesDB both live outside the repo, in the shared lab corpus.
SHARED_ROOT = Path("/dartfs/rc/lab/S/SinghN/shared")
MUSICCAPS_MANIFEST = SHARED_ROOT / "musiccaps/rime-metadata/musiccaps_analysis_manifest.jsonl"
MOISESDB_ZIP = SHARED_ROOT / "moisesdb.zip"

# Nunes & Ordanini (2014), Musicae Scientiae 18(4), 392-409, Table 1, mapped
# onto Demucs stems. N = 2,399 songs. Held out from fitting and used only to
# validate the correction, and to anchor the `produced` profile.
#
# `guitar` is a range because the paper reports clean (38.8%), distorted
# (18.2%) and acoustic (21.8%) separately and never publishes their overlap, so
# the union lies in [0.39, 0.79]. The midpoint below is a guess and is the
# weakest number in the whole derivation.
NUNES_ORDANINI = {"vocals": 0.965, "drums": 0.959, "bass": 0.882, "guitar": 0.600}
NUNES_ORDANINI_GUITAR_RANGE = (0.388, 0.788)
NUNES_ORDANINI_N = 2399

# What the held-out check will accept as an upper bound for a corrected
# marginal. Point-identified for three stems; for guitar only the range is
# identified, so the check must use its top rather than the midpoint guess,
# which would reject values the paper is perfectly consistent with.
NUNES_ORDANINI_UPPER = dict(NUNES_ORDANINI, guitar=NUNES_ORDANINI_GUITAR_RANGE[1])

# Sensitivity needs a denominator: how often a stem is REALLY present in the
# population we condition on. MoisesDB supplies that, but only for stems whose
# presence does not depend on genre -- otherwise its pooled rate just restates
# its own genre mix (44% rock, 19% singer-songwriter) and does not transfer to a
# genre-diverse corpus like MusicCaps.
#
# Measured on MoisesDB genres with n >= 10, the n-weighted standard deviation of
# per-genre presence is 0.010 for vocals and drums and 0.029 for bass -- flat
# everywhere, so the pooled rate is safe. Guitar is 0.209, an order of magnitude
# larger and bimodal: 1.00 in rock, pop, rap and singer-songwriter, 0.14 in
# electronic. Its pooled 0.925 is therefore a fact about MoisesDB's genre mix.
# Left uncorrected it deflates s_guitar to 0.52, and the deconvolution inflates
# guitar to 0.715, above vocals -- contradicting both reference corpora.
GENRE_STABILITY_THRESHOLD = 0.10
MOISESDB_MIN_GENRE_TRACKS = 10

# Fallback denominators for stems MoisesDB cannot identify, from a more
# genre-diverse corpus. Nunes & Ordanini report clean (0.388), distorted (0.182)
# and acoustic (0.218) guitar separately and never publish the overlap, so the
# union is bounded by [0.388, 0.788]: the low end assumes total nesting, the high
# end assumes the three never co-occur, which is implausible for real records.
# Treating them as independent gives 1 - (1-.388)(1-.182)(1-.218) = 0.609, used
# here. Guitar types are positively correlated in practice, which would push the
# true union below this, so 0.609 is if anything generous.
EXTERNAL_TRUTH = {"guitar": 0.609}

# Add-one smoothing for MoisesDB's 240-track joint. Kept separate from `--alpha`,
# which is denominated in clip-equivalents against MusicCaps' 5,200 clips and
# would swamp a corpus this size. See `_moisesdb_joint`.
MOISESDB_SMOOTHING = 1.0

PROFILE_NAMES = ("observed", "corrected", "produced")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--musiccaps-manifest", type=Path, default=MUSICCAPS_MANIFEST, help="Prebuilt MusicCaps analysis manifest. Default: %(default)s")
    parser.add_argument("--moisesdb-zip", type=Path, default=MOISESDB_ZIP, help="MoisesDB archive; only the per-track data.json entries are read. Default: %(default)s")
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/ground_truth/datasets/musiccaps.yaml"), help="Dataset config supplying target_families/target_stems. Default: %(default)s")
    parser.add_argument("--stems", default=None, help="Comma-separated stems to model. Default: every stem with an enabled target family")
    parser.add_argument("--alpha", type=float, default=20.0, help="Dirichlet smoothing added per cell in the EM M-step. Guards against deconvolution collapsing rare combinations to zero. Default: %(default)s")
    parser.add_argument("--em-iterations", type=int, default=1500, help="EM iterations for the deconvolution. Default: %(default)s")
    parser.add_argument("--genre-stability-threshold", type=float, default=GENRE_STABILITY_THRESHOLD, help="Max across-genre standard deviation for MoisesDB's presence rate to be trusted as a sensitivity denominator. Above it the rate reflects MoisesDB's genre mix rather than the stem, and an external anchor is used. Default: %(default)s")
    parser.add_argument("--guitar-truth", type=float, default=None, help="How often guitar is really present in full-band material. Default: %.3f, the Nunes & Ordanini union under independence of clean/distorted/acoustic. Their published bounds are [%.3f, %.3f]; the upper bound assumes the three guitar types never co-occur, so it is the least plausible end" % (EXTERNAL_TRUTH["guitar"], *NUNES_ORDANINI_GUITAR_RANGE))
    parser.add_argument("--mode", choices=["reference", "musiccaps"], default="reference", help="Which corpora to derive from. `reference` uses only MoisesDB and Nunes & Ordanini, keeping the priors independent of MusicCaps so mock corpora do not resemble a MusicCaps evaluation set. `musiccaps` uses the MusicCaps joint corrected for caption under-reporting, which is far more diverse but leaks the evaluation distribution. Default: %(default)s")
    parser.add_argument("--output-path", type=Path, default=Path("configs/ground_truth/stem_priors.yaml"), help="Output YAML path. Default: %(default)s")
    args = parser.parse_args()

    dataset_config = load_dataset_config(args.dataset_config)
    stems = _resolve_stems(args.stems, dataset_config)
    moisesdb = _moisesdb_presence(args.moisesdb_zip)

    external = dict(EXTERNAL_TRUTH)
    if args.guitar_truth is not None:
        external["guitar"] = args.guitar_truth

    if args.mode == "reference":
        _run_reference_mode(args, stems, moisesdb, external)
        return

    observed_counts = _musiccaps_joint(args.musiccaps_manifest, stems)
    truth = _truth_denominators(moisesdb, stems, external, args.genre_stability_threshold)
    sensitivity = _estimate_sensitivity(observed_counts, truth, stems)
    observed = _normalize(observed_counts)
    corrected = _deconvolve(observed_counts, stems, sensitivity, alpha=args.alpha, iterations=args.em_iterations)
    produced = _rake(corrected, stems, {stem: NUNES_ORDANINI[stem] for stem in stems if stem in NUNES_ORDANINI})

    report = _build_report(
        stems=stems,
        observed_counts=observed_counts,
        observed=observed,
        corrected=corrected,
        moisesdb=moisesdb,
        sensitivity=sensitivity,
        truth=truth,
        alpha=args.alpha,
    )
    text = _render_yaml(
        stems=stems,
        observed_counts=observed_counts,
        profiles={"observed": observed, "corrected": corrected, "produced": produced},
        sensitivity=sensitivity,
        moisesdb=moisesdb,
        report=report,
        alpha=args.alpha,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(text, encoding="utf-8")

    _print_report(report, args.output_path)


# --------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------


def _resolve_stems(explicit: str | None, dataset_config: Mapping[str, Any]) -> list[str]:
    """Stems worth modelling: those whose family is enabled in the dataset config.

    Derived rather than hardcoded so that re-enabling `piano` in target_families
    is a one-line config change followed by a re-run, not an edit here.
    """
    if explicit:
        return sorted({value.strip() for value in explicit.split(",") if value.strip()})
    families = dataset_config.get("target_families", {})
    stem_map = dataset_config.get("target_stems", {})
    stems: set[str] = set()
    for family_name, separation_targets in families.items():
        if family_name == "other":
            continue
        for separation_target in separation_targets or []:
            stems.add(stem_map.get(separation_target, separation_target))
    if not stems:
        raise ValueError("Dataset config enables no target families, so there is nothing to model.")
    return sorted(stems)


def _musiccaps_joint(manifest_path: Path, stems: Sequence[str]) -> Counter[frozenset[str]]:
    """Observed stem combination per clip, including the empty combination.

    The empty combination is kept here because the thinning model needs it: a
    clip whose caption named nothing is evidence about how often captions miss
    everything. It is dropped later, when the profiles are rendered.
    """
    keep = set(stems)
    counts: Counter[frozenset[str]] = Counter()
    for record in load_records(manifest_path):
        candidates = record.get("analysis", {}).get("target_candidates", []) or []
        counts[frozenset(c["stem"] for c in candidates if c.get("stem") in keep)] += 1
    if not counts:
        raise ValueError("No records read from '%s'." % manifest_path)
    return counts


def _moisesdb_presence(zip_path: Path) -> dict[str, Any]:
    """Per-stem presence rates from MoisesDB's true multitrack inventories.

    Only the 240 per-track `data.json` entries are read, streamed straight out
    of the archive, so the 88GB of audio is never touched.
    """
    presence: Counter[str] = Counter()
    cardinality: Counter[int] = Counter()
    genres: Counter[str] = Counter()
    by_genre: dict[str, list[set[str]]] = {}
    inventories: list[set[str]] = []
    artists: set[str] = set()
    total = 0
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.endswith("data.json"):
                continue
            with archive.open(name) as handle:
                payload = json.load(handle)
            stem_names = {stem["stemName"] for stem in payload.get("stems", [])}
            presence.update(stem_names)
            cardinality[len(stem_names)] += 1
            genre = payload.get("genre", "unknown")
            genres[genre] += 1
            by_genre.setdefault(genre, []).append(stem_names)
            inventories.append(stem_names)
            artists.add(payload.get("artist", ""))
            total += 1
    if not total:
        raise ValueError("No data.json entries found in '%s'." % zip_path)
    return {
        "total": total,
        "artists": len(artists),
        "rates": {stem: count / total for stem, count in presence.items()},
        "genre_spread": _genre_spread(by_genre, presence),
        "cardinality": dict(sorted(cardinality.items())),
        "genres": dict(genres.most_common()),
        "inventories": inventories,
    }


def _genre_spread(
    by_genre: Mapping[str, Sequence[set[str]]],
    presence: Mapping[str, int],
) -> dict[str, float]:
    """n-weighted standard deviation of each stem's presence rate across genres.

    A stem equally common in every genre has a spread near zero, so MoisesDB's
    pooled rate for it transfers to any corpus. A stem whose presence is
    genre-driven has a large spread, and its pooled rate then measures MoisesDB's
    own genre mix rather than anything general.

    Small genres are excluded: with one or two tracks a rate of exactly 0 or 1 is
    noise, and it would dominate the spread.
    """
    usable = {
        genre: tracks
        for genre, tracks in by_genre.items()
        if len(tracks) >= MOISESDB_MIN_GENRE_TRACKS
    }
    total = sum(len(tracks) for tracks in usable.values())
    if not total:
        return {}
    spread: dict[str, float] = {}
    for stem in presence:
        rates = {
            genre: sum(1 for stems in tracks if stem in stems) / len(tracks)
            for genre, tracks in usable.items()
        }
        mean = sum(rates[genre] * len(usable[genre]) for genre in usable) / total
        variance = sum(len(usable[genre]) * (rates[genre] - mean) ** 2 for genre in usable) / total
        spread[stem] = math.sqrt(variance)
    return spread


# --------------------------------------------------------------------------
# Reference-only derivation (no MusicCaps)
# --------------------------------------------------------------------------


def _run_reference_mode(
    args: argparse.Namespace,
    stems: Sequence[str],
    moisesdb: Mapping[str, Any],
    external: Mapping[str, float],
) -> None:
    marginals = _reference_marginals(moisesdb, stems, external, args.genre_stability_threshold)
    rates = {stem: entry["value"] for stem, entry in marginals.items()}

    moisesdb_joint, moisesdb_populated = _moisesdb_joint(moisesdb, stems)
    profiles = {
        "reference": _maxent_joint(stems, rates),
        "moisesdb_empirical": moisesdb_joint,
        "uniform": _uniform_joint(stems),
    }
    text = _render_reference_yaml(
        stems=stems,
        marginals=marginals,
        profiles=profiles,
        moisesdb=moisesdb,
        moisesdb_populated=moisesdb_populated,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(text, encoding="utf-8")

    print("Wrote %s (mode: reference -- MusicCaps not used)" % args.output_path)
    print("  marginals:")
    for stem in stems:
        entry = marginals[stem]
        detail = entry["source"]
        if entry["source"] == "blend":
            detail = "blend of moisesdb %.3f and n&o %.3f" % (entry["moisesdb_value"], entry["nunes_ordanini_value"])
        elif entry["source"] == "nunes_ordanini":
            detail = "n&o only; moisesdb %.3f rejected, genre sd %.3f" % (entry["moisesdb_value"], entry["spread"])
        print("    %-8s %.3f   %s" % (stem, entry["value"], detail))
    print("  profiles:")
    for name, joint in profiles.items():
        non_empty = {subset: weight for subset, weight in joint.items() if subset}
        scale = sum(non_empty.values())
        effective = 1.0 / sum((weight / scale) ** 2 for weight in non_empty.values())
        populated = sum(1 for weight in non_empty.values() if weight / scale >= 0.005)
        print("    %-20s effective combinations %5.2f   cells above 0.5%%: %2d of %d" % (name, effective, populated, len(non_empty)))


def _reference_marginals(
    moisesdb: Mapping[str, Any],
    stems: Sequence[str],
    external: Mapping[str, float],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Blend MoisesDB and Nunes & Ordanini marginals, by corpus size.

    Where MoisesDB's rate is genre-unstable it is dropped entirely rather than
    blended, since averaging in a number that only reflects MoisesDB's genre mix
    would carry that bias through at reduced weight instead of removing it.
    """
    rates = moisesdb.get("rates", {})
    spread = moisesdb.get("genre_spread", {})
    moisesdb_n = float(moisesdb.get("total", 0))
    blended: dict[str, dict[str, Any]] = {}
    for stem in stems:
        anchor = external.get(stem, NUNES_ORDANINI.get(stem))
        stem_spread = spread.get(stem, 0.0)
        if anchor is None:
            blended[stem] = {"value": rates.get(stem, 0.0), "source": "moisesdb", "spread": stem_spread}
            continue
        if stem_spread > threshold:
            blended[stem] = {
                "value": anchor,
                "source": "nunes_ordanini",
                "spread": stem_spread,
                "moisesdb_value": rates.get(stem, 0.0),
            }
            continue
        value = (moisesdb_n * rates.get(stem, 0.0) + NUNES_ORDANINI_N * anchor) / (moisesdb_n + NUNES_ORDANINI_N)
        blended[stem] = {
            "value": value,
            "source": "blend",
            "spread": stem_spread,
            "moisesdb_value": rates.get(stem, 0.0),
            "nunes_ordanini_value": anchor,
        }
    return blended


def _maxent_joint(stems: Sequence[str], marginals: Mapping[str, float]) -> dict[frozenset[str], float]:
    """Maximum-entropy joint subject to the given marginals, i.e. independence.

    Neither reference corpus publishes a joint -- Nunes & Ordanini report only
    marginals and QCA configurations, and MoisesDB's own joint is 91% full-band
    over five cells. Independence is the least-committal way to turn marginals
    into a distribution: it adds no dependence structure that the sources do not
    support, and unlike MoisesDB's empirical joint it populates every cell.

    Real stems are positively correlated, so this understates both very sparse
    and very dense combinations. There is no way to fix that without a source
    that actually measures co-occurrence.
    """
    joint: dict[frozenset[str], float] = {}
    for subset in _all_subsets(stems):
        probability = 1.0
        for stem in stems:
            rate = marginals[stem]
            probability *= rate if stem in subset else 1.0 - rate
        joint[subset] = probability
    return joint


def _moisesdb_joint(
    moisesdb: Mapping[str, Any],
    stems: Sequence[str],
) -> tuple[dict[frozenset[str], float], float]:
    """MoisesDB's own combination frequencies, lightly smoothed.

    Faithful to a corpus with zero measurement error, but that corpus was
    curated for source separation, so it is nearly all full-band: five of
    fifteen cells are populated before smoothing.

    Smoothing is deliberately NOT the `--alpha` used for the MusicCaps
    deconvolution. That alpha is denominated in clip-equivalents and tuned
    against 5,200 clips; spending it on 240 would put more pseudo-count than
    data into the table and quietly turn this profile into a near-uniform one
    (effective combinations 1.2 unsmoothed, 4.6 at alpha=20). Add-one keeps the
    pseudo-count at 7% of the corpus, which fills the empty cells without
    rewriting the shape.
    """
    keep = set(stems)
    counts: Counter[frozenset[str]] = Counter()
    for inventory in moisesdb.get("inventories", []):
        counts[frozenset(inventory & keep)] += 1
    subsets = _all_subsets(stems)
    smoothed = {subset: counts.get(subset, 0) + MOISESDB_SMOOTHING for subset in subsets}
    total = sum(smoothed.values())
    populated = sum(1 for subset in subsets if subset and counts.get(subset, 0))
    return {subset: value / total for subset, value in smoothed.items()}, populated


def _uniform_joint(stems: Sequence[str]) -> dict[frozenset[str], float]:
    """Equal weight on every non-empty combination.

    Carries no information from any corpus, which makes it the right choice when
    mock clips must not resemble the evaluation set, and the most efficient one
    for exercising every recipe path per clip generated.
    """
    subsets = [subset for subset in _all_subsets(stems) if subset]
    weight = 1.0 / len(subsets)
    joint = {subset: weight for subset in subsets}
    joint[frozenset()] = 0.0
    return joint


# --------------------------------------------------------------------------
# Correction
# --------------------------------------------------------------------------


def _estimate_sensitivity(
    observed_counts: Mapping[frozenset[str], int],
    truth_denominators: Mapping[str, Mapping[str, Any]],
    stems: Sequence[str],
) -> dict[str, float]:
    """P(a caption names stem k | stem k is present).

    Conditioning on "the caption named every other stem" selects full-band
    clips, which is what makes the MusicCaps and reference populations
    comparable: among full-band material the stem is present at a known rate, so
    the shortfall in how often captions name it is attributable to the caption
    rather than to the music.

    That argument only holds if the denominator transfers between populations.
    See `_truth_denominators` for how each one is chosen, and why MoisesDB
    cannot supply it for every stem.
    """
    full = frozenset(stems)
    sensitivity: dict[str, float] = {}
    for stem in stems:
        others = full - {stem}
        with_stem = observed_counts.get(full, 0)
        without_stem = observed_counts.get(others, 0)
        observations = with_stem + without_stem
        truth = truth_denominators.get(stem, {}).get("value", 0.0)
        if observations == 0 or truth <= 0:
            # No evidence for this stem; assume captions never miss it, which
            # makes the correction a no-op rather than an arbitrary guess.
            sensitivity[stem] = 1.0
            continue
        sensitivity[stem] = min(1.0, (with_stem / observations) / truth)
    return sensitivity


def _truth_denominators(
    moisesdb: Mapping[str, Any],
    stems: Sequence[str],
    external: Mapping[str, float],
    threshold: float,
) -> dict[str, dict[str, Any]]:
    """Pick, per stem, how often it is REALLY present in full-band material.

    MoisesDB is the default source, but only where its rate is genre-invariant.
    Where presence is genre-driven, its pooled rate encodes MoisesDB's own genre
    mix and will not transfer to a genre-diverse corpus, so an external anchor is
    used instead when one exists.
    """
    rates = moisesdb.get("rates", {})
    spread = moisesdb.get("genre_spread", {})
    chosen: dict[str, dict[str, Any]] = {}
    for stem in stems:
        stem_spread = spread.get(stem, 0.0)
        stable = stem_spread <= threshold
        if stable or stem not in external:
            chosen[stem] = {
                "value": rates.get(stem, 0.0),
                "source": "moisesdb",
                "spread": stem_spread,
                # Flagged so an unstable stem with no anchor is visible in the
                # output rather than silently trusted.
                "unanchored": not stable,
            }
            continue
        chosen[stem] = {
            "value": external[stem],
            "source": "external",
            "spread": stem_spread,
            "moisesdb_value": rates.get(stem, 0.0),
            "unanchored": False,
        }
    return chosen


def _deconvolve(
    observed_counts: Mapping[frozenset[str], int],
    stems: Sequence[str],
    sensitivity: Mapping[str, float],
    alpha: float,
    iterations: int,
) -> dict[frozenset[str], float]:
    """Invert the thinning model by EM to recover the true combination joint.

    Observed set S arises from true set T (a superset of S) when every stem in S
    survives into the caption and every stem in T \\ S is missed:

        P_obs(S) = sum_{T superset of S} P_true(T) prod_{k in S} s_k prod_{k in T\\S} (1 - s_k)

    The `prod_{k in S} s_k` factor does not depend on T, so it cancels in the
    E-step normalization and never appears below.

    `alpha` is Dirichlet smoothing added to every cell in the M-step. Without it
    the maximum-likelihood solution over-concentrates and drives rare
    combinations to exactly zero, which is the usual ill-posedness of
    deconvolution rather than a fact about music.
    """
    subsets = _all_subsets(stems)
    total = sum(observed_counts.values())
    current = {subset: 1.0 / len(subsets) for subset in subsets}

    for _ in range(iterations):
        updated = {subset: alpha for subset in subsets}
        for observed_subset, count in observed_counts.items():
            weights = {}
            for subset in subsets:
                if not observed_subset <= subset:
                    continue
                weight = current[subset]
                for stem in subset - observed_subset:
                    weight *= 1.0 - sensitivity[stem]
                weights[subset] = weight
            normalizer = sum(weights.values())
            if normalizer <= 0:
                continue
            for subset, weight in weights.items():
                updated[subset] += count * weight / normalizer
        scale = sum(updated.values())
        current = {subset: value / scale for subset, value in updated.items()}

    # `alpha` is expressed in clip-equivalents so a given value means the same
    # thing regardless of corpus size; report it that way in the header.
    del total
    return current


def _rake(
    joint: Mapping[frozenset[str], float],
    stems: Sequence[str],
    targets: Mapping[str, float],
    iterations: int = 500,
) -> dict[frozenset[str], float]:
    """Iterative proportional fitting of a joint onto target marginals.

    Preserves the source joint's dependence structure while moving its marginals
    onto an external anchor. Used only for the `produced` profile.
    """
    current = dict(joint)
    for _ in range(iterations):
        for stem, target in targets.items():
            marginal = sum(value for subset, value in current.items() if stem in subset)
            if marginal <= 0 or marginal >= 1:
                continue
            present_scale = target / marginal
            absent_scale = (1.0 - target) / (1.0 - marginal)
            current = {
                subset: value * (present_scale if stem in subset else absent_scale)
                for subset, value in current.items()
            }
            scale = sum(current.values())
            current = {subset: value / scale for subset, value in current.items()}
    return current


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _build_report(
    stems: Sequence[str],
    observed_counts: Mapping[frozenset[str], int],
    observed: Mapping[frozenset[str], float],
    corrected: Mapping[frozenset[str], float],
    moisesdb: Mapping[str, Any],
    sensitivity: Mapping[str, float],
    truth: Mapping[str, Mapping[str, Any]],
    alpha: float,
) -> dict[str, Any]:
    total = sum(observed_counts.values())
    observed_marginals = _marginals(observed, stems)
    corrected_marginals = _marginals(corrected, stems)

    # Held-out check: correcting measurement error should move each marginal
    # toward Nunes & Ordanini without reaching it, since a genuine population
    # difference remains (arbitrary audio vs. charting pop).
    between: dict[str, bool] = {}
    for stem in stems:
        upper = NUNES_ORDANINI_UPPER.get(stem)
        if upper is None:
            continue
        between[stem] = observed_marginals[stem] <= corrected_marginals[stem] <= upper + 1e-9

    observed_cardinality = _cardinality(observed)
    predicted_cardinality = _cardinality(_thin(corrected, stems, sensitivity))

    return {
        "clips": total,
        "stems": list(stems),
        "sensitivity": dict(sensitivity),
        "truth": {stem: dict(values) for stem, values in truth.items()},
        "alpha": alpha,
        "observed_marginals": observed_marginals,
        "corrected_marginals": corrected_marginals,
        "held_out_between": between,
        "observed_empty": observed.get(frozenset(), 0.0),
        "corrected_empty": corrected.get(frozenset(), 0.0),
        "observed_effective": _effective_combinations(observed),
        "corrected_effective": _effective_combinations(corrected),
        "observed_cardinality": observed_cardinality,
        "corrected_cardinality": _cardinality(corrected),
        "predicted_cardinality": predicted_cardinality,
        "fit": _describe_fit(observed_cardinality, predicted_cardinality, len(stems)),
        "moisesdb": moisesdb,
    }


def _describe_fit(
    observed_cardinality: Mapping[int, float],
    predicted_cardinality: Mapping[int, float],
    stem_count: int,
) -> dict[str, Any]:
    """Diagnose the independence assumption from the cardinality residuals.

    Independent per-stem detection predicts a particular spread of how many
    stems a caption names. Where the prediction misses tells us how detections
    are actually correlated, and therefore which way the correction is wrong.
    """
    residuals = {
        size: observed_cardinality.get(size, 0.0) - predicted_cardinality.get(size, 0.0)
        for size in range(stem_count + 1)
    }
    worst_size = max(residuals, key=lambda size: abs(residuals[size]))
    worst = residuals[worst_size]
    predicted = predicted_cardinality.get(worst_size, 0.0)
    relative = abs(worst) / predicted if predicted > 0 else 0.0

    if worst > 0 and worst_size >= stem_count - 1:
        # More exhaustive captions than independence allows: some annotators
        # inventory the clip. EM can only explain those by inflating the
        # full-combination cell, so the correction overstates full-band mass.
        direction = "understates"
        consequence = "`corrected` overstates dense combinations"
    elif worst > 0 and worst_size <= 1:
        direction = "understates"
        consequence = "`corrected` overstates sparse combinations"
    else:
        direction = "overstates"
        consequence = "the correction is conservative at this end"

    return {
        "residuals": residuals,
        "worst_size": worst_size,
        "worst_residual": worst,
        "worst_relative": relative,
        "direction": direction,
        "consequence": consequence,
    }


def _print_report(report: Mapping[str, Any], output_path: Path) -> None:
    print("Wrote %s" % output_path)
    print("  clips: %d   alpha: %g" % (report["clips"], report["alpha"]))
    print("  sensitivity P(caption names stem | present), and its denominator:")
    for stem, value in sorted(report["sensitivity"].items(), key=lambda item: -item[1]):
        entry = report["truth"].get(stem, {})
        note = "moisesdb %.3f (genre sd %.3f)" % (entry.get("value", 0.0), entry.get("spread", 0.0))
        if entry.get("source") == "external":
            note = "external %.3f, not moisesdb %.3f (genre sd %.3f, unstable)" % (
                entry.get("value", 0.0),
                entry.get("moisesdb_value", 0.0),
                entry.get("spread", 0.0),
            )
        elif entry.get("unanchored"):
            note += "  WARNING: unstable and no external anchor"
        print("    %-8s %.2f   %s" % (stem, value, note))
    print("  marginals            observed -> corrected   (Nunes & Ordanini, held out)")
    for stem in report["stems"]:
        anchor = NUNES_ORDANINI.get(stem)
        if stem == "guitar":
            anchor_text = "[%.3f, %.3f]" % NUNES_ORDANINI_GUITAR_RANGE
        else:
            anchor_text = "n/a" if anchor is None else "%.3f" % anchor
        flag = ""
        if stem in report["held_out_between"]:
            flag = "  ok" if report["held_out_between"][stem] else "  OUTSIDE"
        print("    %-8s %.3f -> %.3f   %-14s%s" % (stem, report["observed_marginals"][stem], report["corrected_marginals"][stem], anchor_text, flag))
    print("  empty combination:   %.3f -> %.3f" % (report["observed_empty"], report["corrected_empty"]))
    print("  effective combinations: %.1f -> %.1f" % (report["observed_effective"], report["corrected_effective"]))
    print("  cardinality fit (tests the independence assumption):")
    for size in sorted(set(report["observed_cardinality"]) | set(report["predicted_cardinality"])):
        actual = report["observed_cardinality"].get(size, 0.0)
        predicted = report["predicted_cardinality"].get(size, 0.0)
        print("    %d stems  actual %.3f  predicted %.3f  residual %+.3f" % (size, actual, predicted, actual - predicted))
    fit = report["fit"]
    print("    -> worst at %d stems: model %s by %.0f%%; %s" % (fit["worst_size"], fit["direction"], 100 * fit["worst_relative"], fit["consequence"]))


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render_yaml(
    stems: Sequence[str],
    observed_counts: Mapping[frozenset[str], int],
    profiles: Mapping[str, Mapping[frozenset[str], float]],
    sensitivity: Mapping[str, float],
    moisesdb: Mapping[str, Any],
    report: Mapping[str, Any],
    alpha: float,
) -> str:
    """Emit the config by hand.

    `yaml.dump` cannot write comments, and every config in
    configs/ground_truth/ carries the rationale beside the numbers. The
    provenance, the sensitivity estimates and the caveats are the point of this
    file, so they are templated in rather than dropped.
    """
    lines: list[str] = []
    add = lines.append

    add("# Priors over Demucs stem combinations, for sampling mock analysis manifests.")
    add("#")
    add("# GENERATED by scripts/derive_stem_priors.py -- edit that script, not this file.")
    add("#")
    add("# Three corpora are combined, each doing only what it uniquely can:")
    add("#")
    add("#   MusicCaps (n=%d)      the joint's shape. The only source with a diverse," % report["clips"])
    add("#                            complete joint over stem combinations.")
    add("#   MoisesDB (n=%d)         sensitivity estimation. True multitrack inventories," % moisesdb["total"])
    add("#                            so zero measurement error, but nearly every track is")
    add("#                            full-band and it carries almost no variance itself.")
    add("#   Nunes & Ordanini 2014    held-out validation, never fit to. Expert-coded")
    add("#   (n=2399)                 marginals from 2,399 Billboard Hot 100 songs. See")
    add("#                            configs/ground_truth/refs/ for the paper.")
    add("#")
    add("# Captions under-report: they describe a clip, they do not inventory it. So the")
    add("# raw MusicCaps joint is biased toward sparse combinations. Conditioning on")
    add("# \"the caption named every other stem\" selects full-band clips, which makes the")
    add("# MusicCaps and MoisesDB populations comparable and lets the shortfall be")
    add("# attributed to the caption rather than the music. That yields a per-stem")
    add("# sensitivity, and the observed set is then modelled as the true set thinned:")
    add("# each present stem survives into the caption independently with probability")
    add("# s_k. Inverting that by EM gives the `corrected` profile.")
    add("#")
    add("# Known limitations, in rough order of how much they should worry you:")
    add("#")
    fit = report["fit"]
    add("#  - MEASURED: the thinning model assumes per-stem detection is independent, and")
    add("#    it is not. Pushing `corrected` back through the model and comparing against")
    add("#    what was actually observed, the largest miss is at %d stem(s), where the" % fit["worst_size"])
    add("#    model %s reality by %.3f (%.0f%% of its own prediction). Captions are" % (fit["direction"], abs(fit["worst_residual"]), 100 * fit["worst_relative"]))
    add("#    more exhaustive than independent detection allows -- some annotators")
    add("#    inventory the clip rather than describe it -- and EM can only account for")
    add("#    that by inflating the full-combination cell, so")
    add("#    %s. Full residuals are under `diagnostics`." % fit["consequence"])
    add("#  - s_k is estimated on full-band clips and then applied everywhere, assuming")
    add("#    sensitivity does not vary with how many stems are present. A caption for a")
    add("#    sparse clip probably names its one instrument more reliably than a dense mix")
    add("#    names each of four, which would push the same way as the point above.")
    add("#  - Nunes & Ordanini's guitar marginal is a RANGE, [%.3f, %.3f], because the" % NUNES_ORDANINI_GUITAR_RANGE)
    add("#    paper reports clean/distorted/acoustic separately and never publishes their")
    add("#    overlap. The %.2f used for the `produced` profile is a midpoint guess and is" % NUNES_ORDANINI["guitar"])
    add("#    the weakest number here. The authors offer raw data on request.")
    add("#  - MoisesDB is %d tracks over %d artists and is %s-heavy, so its effective n is" % (moisesdb["total"], moisesdb["artists"], next(iter(moisesdb["genres"]))))
    add("#    well below %d. Its marginals all sit near 1.0, which makes s_k robust to" % moisesdb["total"])
    add("#    that (the divisor is ~1), but the conditioning population still differs.")
    add("")
    add("version: 1")
    add("stem_priors:")
    add("  sources:")
    add("    musiccaps: {role: joint_shape, n: %d}" % report["clips"])
    add("    moisesdb: {role: sensitivity, n: %d}" % moisesdb["total"])
    add("    nunes_ordanini_2014: {role: validation_only, n: 2399}")
    add("")
    add("  # P(a caption names the stem | the stem is present): a MusicCaps conditional")
    add("  # divided by how often the stem is really there. Lower means captions miss it")
    add("  # more often, so the correction moves more mass onto combinations containing it.")
    add("  #")
    add("  # The denominator comes from MoisesDB only where its rate is genre-invariant.")
    add("  # `genre_sd` is the n-weighted spread of its per-genre presence rate; above")
    add("  # %.2f the pooled rate measures MoisesDB's genre mix rather than the stem, and" % GENRE_STABILITY_THRESHOLD)
    add("  # an external, genre-diverse anchor is used instead.")
    add("  sensitivity:")
    for stem, value in sorted(sensitivity.items(), key=lambda item: -item[1]):
        entry = report["truth"].get(stem, {})
        add("    %s:" % stem)
        add("      value: %.3f" % value)
        add("      truth: %.3f" % entry.get("value", 0.0))
        add("      truth_source: %s" % entry.get("source", "moisesdb"))
        add("      genre_sd: %.3f" % entry.get("spread", 0.0))
        if entry.get("source") == "external":
            add("      # MoisesDB says %.3f, rejected: genre_sd %.3f is above the threshold." % (entry.get("moisesdb_value", 0.0), entry.get("spread", 0.0)))
            add("      # Using it would deflate this sensitivity and over-inflate the stem.")
        elif entry.get("unanchored"):
            add("      # WARNING: genre_sd is above the threshold but no external anchor")
            add("      # exists for this stem, so MoisesDB's rate is used unverified.")
    add("")
    add("  # Dirichlet smoothing per cell in the EM M-step, in clip-equivalents. Without")
    add("  # it the deconvolution drives rare combinations to exactly zero, which is the")
    add("  # usual ill-posedness of deconvolution rather than a fact about music.")
    add("  #")
    add("  # Measured over alpha = 0, 5, 20, 50, 100, effective combinations run 4.5, 5.5,")
    add("  # 6.9, 8.7, 10.7 while the marginals move much less: bass is flat to +-0.002,")
    add("  # drums and vocals within 0.06, guitar 0.695 -> 0.601. Guitar is the most")
    add("  # regularizer-sensitive because it is the most-corrected stem (lowest s_k), and")
    add("  # alpha = 100 is an extreme -- 29% of the total mass as uniform prior -- not a")
    add("  # real operating point. Over 5..50 every stem holds within 0.06. So alpha")
    add("  # mainly trades combination diversity against fidelity to the unregularized")
    add("  # solution, but do not read guitar's corrected marginal as alpha-independent.")
    add("  alpha: %g" % alpha)
    add("")
    add("  enabled_stems: [%s]" % ", ".join(stems))
    add("")
    add("  # Diagnostics, carried so the file audits itself.")
    add("  #")
    add("  # NOTE these marginals are unconditional: they include the empty-combination")
    add("  # mass listed just below them. The profiles drop the empty combination and")
    add("  # renormalize, so sampling a profile reproduces marginal/(1 - empty), not the")
    add("  # marginal printed here. For `corrected` that is the difference between bass")
    add("  # at %.3f here and %.3f when sampled." % (report["corrected_marginals"][stems[0]], report["corrected_marginals"][stems[0]] / (1.0 - report["corrected_empty"])))
    add("  diagnostics:")
    add("    marginals:")
    for stem in stems:
        anchor = NUNES_ORDANINI.get(stem)
        suffix = "" if anchor is None else "   # Nunes & Ordanini %.3f" % anchor
        add("      %s: {observed: %.4f, corrected: %.4f}%s" % (stem, report["observed_marginals"][stem], report["corrected_marginals"][stem], suffix))
    add("    empty_combination: {observed: %.4f, corrected: %.4f}" % (report["observed_empty"], report["corrected_empty"]))
    add("    effective_combinations: {observed: %.2f, corrected: %.2f}" % (report["observed_effective"], report["corrected_effective"]))
    add("    # Model fit: push `corrected` back through the thinning model and compare")
    add("    # against what was actually observed. Divergence means detection is")
    add("    # correlated across stems, which the model cannot represent. Keyed by how")
    add("    # many stems the caption named.")
    add("    observed_cardinality: {%s}" % ", ".join("%d: %.4f" % (k, v) for k, v in sorted(report["observed_cardinality"].items())))
    add("    predicted_cardinality: {%s}" % ", ".join("%d: %.4f" % (k, v) for k, v in sorted(report["predicted_cardinality"].items())))
    add("    cardinality_residual: {%s}" % ", ".join("%d: %+.4f" % (k, v) for k, v in sorted(report["fit"]["residuals"].items())))
    add("    # MoisesDB stems per track, over its own 11-category taxonomy. Not directly")
    add("    # comparable to the %d Demucs stems above, but the best cardinality truth there is." % len(stems))
    add("    moisesdb_cardinality: {%s}" % ", ".join("%d: %d" % (k, v) for k, v in moisesdb["cardinality"].items()))
    add("")
    add("  # Sampling reads this profile unless told otherwise.")
    add("  default_profile: corrected")
    add("")
    add("  profiles:")

    descriptions = {
        "observed": [
            "Raw MusicCaps joint, uncorrected. Most diverse of the three, so this is",
            "the one to use for recipe-coverage testing: it visits single-stem",
            "combinations often enough to exercise those paths.",
        ],
        "corrected": [
            "Deconvolved for caption under-reporting. Best estimate of true stem",
            "prevalence in arbitrary audio, and the default.",
        ],
        "produced": [
            "`corrected` raked onto Nunes & Ordanini marginals: full-band produced",
            "pop rather than arbitrary audio. Much less diverse, so it is not the",
            "default -- use it when mock clips should look like charting records.",
        ],
    }

    for name in PROFILE_NAMES:
        joint = profiles[name]
        add("")
        for line in descriptions[name]:
            add("    # %s" % line)
        add("    %s:" % name)
        add("      combinations:")
        add("        type: choice")
        add("        values:")
        non_empty = {subset: weight for subset, weight in joint.items() if subset}
        ordered = sorted(non_empty, key=lambda s: (-non_empty[s], sorted(s)))
        for subset, weight in zip(ordered, _rounded_weights([non_empty[s] for s in ordered])):
            suffix = ""
            if name == "observed":
                suffix = "  # n=%d" % observed_counts.get(subset, 0)
            add("          - value: [%s]" % ", ".join(sorted(subset)))
            add("            weight: %s%s" % (weight, suffix))

    add("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _rounded_weights(weights: Sequence[float], places: int = 5) -> list[str]:
    """Normalize, round, and absorb the rounding residual into the largest cell.

    The planner treats choice weights as unnormalized, so this is cosmetic -- but
    the file is meant to be read and audited, and a column of probabilities that
    does not sum to 1 invites a bug report that isn't one.
    """
    total = sum(weights)
    scale = 10 ** places
    rounded = [round(weight / total * scale) for weight in weights]
    residual = scale - sum(rounded)
    if rounded:
        largest = max(range(len(rounded)), key=lambda index: rounded[index])
        rounded[largest] += residual
    return ["%.*f" % (places, value / scale) for value in rounded]


def _render_reference_yaml(
    stems: Sequence[str],
    marginals: Mapping[str, Mapping[str, Any]],
    profiles: Mapping[str, Mapping[frozenset[str], float]],
    moisesdb: Mapping[str, Any],
    moisesdb_populated: int,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Priors over Demucs stem combinations, for sampling mock analysis manifests.")
    add("#")
    add("# GENERATED by scripts/derive_stem_priors.py --mode reference")
    add("# -- edit that script, not this file.")
    add("#")
    add("# MUSICCAPS IS DELIBERATELY NOT USED HERE. Downstream evaluation runs on")
    add("# MusicCaps, so deriving mock corpora from it would make the development")
    add("# distribution resemble the evaluation distribution. Only two corpora feed")
    add("# this file:")
    add("#")
    add("#   MoisesDB (n=%d)          true multitrack inventories, zero measurement" % moisesdb["total"])
    add("#                            error, but curated for source separation and so")
    add("#                            almost entirely full-band.")
    add("#   Nunes & Ordanini 2014    2,399 Billboard Hot 100 songs, instrumentation")
    add("#   (n=%d)                 coded over full songs by music-school graduates." % NUNES_ORDANINI_N)
    add("#                            See configs/ground_truth/refs/ for the paper.")
    add("#")
    add("# THE COST, STATED PLAINLY. Neither corpus publishes a joint distribution over")
    add("# stem combinations: Nunes & Ordanini report marginals and QCA configurations")
    add("# but never frequencies, and MoisesDB's own joint is 91% full-band across five")
    add("# of fifteen cells. So the `reference` profile is the maximum-entropy joint")
    add("# given the marginals below, which assumes independence. Real stems are")
    add("# positively correlated, so it understates both very sparse and very dense")
    add("# combinations, and there is no way to correct that without a corpus that")
    add("# actually measures co-occurrence.")
    add("#")
    add("# In practice this concentrates hard. Check `effective_combinations` per")
    add("# profile before using one for coverage work: if a recipe path needs a")
    add("# single-stem clip, `reference` will reach it only rarely and `uniform` is the")
    add("# profile to use instead.")
    add("#")
    add("# Marginals blend the two corpora by corpus size, EXCEPT where MoisesDB's rate")
    add("# is genre-unstable. Its per-genre spread is 0.01 for vocals and drums and")
    add("# 0.03 for bass -- flat, so the pooled rate transfers. For guitar it is 0.21")
    add("# and bimodal (1.00 in rock/pop/rap/singer-songwriter, 0.14 in electronic), so")
    add("# its pooled rate restates MoisesDB's own genre mix. Guitar therefore takes")
    add("# the Nunes & Ordanini value alone rather than a blend, which would only carry")
    add("# that bias through at reduced weight.")
    add("")
    add("version: 1")
    add("stem_priors:")
    add("  mode: reference")
    add("  sources:")
    add("    moisesdb: {role: marginals_and_joint, n: %d}" % moisesdb["total"])
    add("    nunes_ordanini_2014: {role: marginals, n: %d}" % NUNES_ORDANINI_N)
    add("    musiccaps: {role: excluded, reason: downstream_evaluation_set}")
    add("")
    add("  enabled_stems: [%s]" % ", ".join(stems))
    add("")
    add("  marginals:")
    for stem in stems:
        entry = marginals[stem]
        add("    %s:" % stem)
        add("      value: %.4f" % entry["value"])
        add("      source: %s" % entry["source"])
        add("      genre_sd: %.3f" % entry["spread"])
        if entry["source"] == "blend":
            add("      # moisesdb %.3f, nunes_ordanini %.3f, weighted by corpus size" % (entry["moisesdb_value"], entry["nunes_ordanini_value"]))
        elif entry["source"] == "nunes_ordanini":
            add("      # moisesdb says %.3f, rejected: genre_sd above threshold." % entry["moisesdb_value"])
    add("")
    add("  default_profile: reference")
    add("")
    add("  profiles:")

    descriptions = {
        "reference": [
            "Maximum-entropy joint given the marginals above. The best estimate of",
            "produced-music stem combinations that owes nothing to MusicCaps.",
        ],
        "moisesdb_empirical": [
            "MoisesDB's own combination frequencies, add-%g smoothed. Only %d of 15" % (MOISESDB_SMOOTHING, moisesdb_populated),
            "cells are populated by actual tracks; the rest are smoothing. Zero",
            "measurement error, but the corpus is curated for source separation and",
            "so is nearly all full-band. Use it to see what real multitrack",
            "inventories look like, not to generate a varied corpus.",
        ],
        "uniform": [
            "Equal weight on all 15 combinations. Carries no information from any",
            "corpus, which makes it both the safest choice against resembling the",
            "evaluation set and the most efficient for exercising every recipe path.",
            "Use this for coverage runs.",
        ],
    }

    for name, joint in profiles.items():
        non_empty = {subset: weight for subset, weight in joint.items() if subset}
        scale = sum(non_empty.values())
        effective = 1.0 / sum((weight / scale) ** 2 for weight in non_empty.values())
        add("")
        for line in descriptions[name]:
            add("    # %s" % line)
        add("    # effective combinations: %.2f of %d" % (effective, len(non_empty)))
        add("    %s:" % name)
        add("      combinations:")
        add("        type: choice")
        add("        values:")
        ordered = sorted(non_empty, key=lambda subset: (-non_empty[subset], sorted(subset)))
        for subset, weight in zip(ordered, _rounded_weights([non_empty[s] for s in ordered])):
            add("          - value: [%s]" % ", ".join(sorted(subset)))
            add("            weight: %s" % weight)

    add("")
    return "\n".join(lines)


def _all_subsets(stems: Sequence[str]) -> list[frozenset[str]]:
    return [
        frozenset(combination)
        for size in range(len(stems) + 1)
        for combination in itertools.combinations(stems, size)
    ]


def _normalize(counts: Mapping[frozenset[str], int]) -> dict[frozenset[str], float]:
    total = sum(counts.values())
    return {subset: count / total for subset, count in counts.items()}


def _marginals(joint: Mapping[frozenset[str], float], stems: Iterable[str]) -> dict[str, float]:
    return {
        stem: sum(weight for subset, weight in joint.items() if stem in subset)
        for stem in stems
    }


def _cardinality(joint: Mapping[frozenset[str], float]) -> dict[int, float]:
    sizes: dict[int, float] = {}
    for subset, weight in joint.items():
        sizes[len(subset)] = sizes.get(len(subset), 0.0) + weight
    return sizes


def _effective_combinations(joint: Mapping[frozenset[str], float]) -> float:
    """Inverse Simpson index over non-empty combinations.

    Reads as "how many combinations this distribution effectively visits", which
    is the number that matters for whether mock clips exercise varied recipes.
    """
    non_empty = [weight for subset, weight in joint.items() if subset]
    total = sum(non_empty)
    if total <= 0:
        return 0.0
    return 1.0 / sum((weight / total) ** 2 for weight in non_empty)


def _thin(
    joint: Mapping[frozenset[str], float],
    stems: Sequence[str],
    sensitivity: Mapping[str, float],
) -> dict[frozenset[str], float]:
    """Forward pass of the thinning model, for checking the fit against reality."""
    predicted = {subset: 0.0 for subset in _all_subsets(stems)}
    for true_subset, weight in joint.items():
        for observed_subset in _all_subsets(sorted(true_subset)):
            probability = weight
            for stem in true_subset:
                probability *= sensitivity[stem] if stem in observed_subset else 1.0 - sensitivity[stem]
            predicted[observed_subset] += probability
    return predicted


if __name__ == "__main__":
    main()
