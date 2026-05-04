#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VALIDATION_ROOT="${VALIDATION_ROOT:-${REPO_ROOT}/derived/validation/mtg_jamendo_10}"
ALL_PLANS_PATH="${ALL_PLANS_PATH:-${VALIDATION_ROOT}/all_plans.jsonl}"
SUBSAMPLED_PLANS_PATH="${SUBSAMPLED_PLANS_PATH:-${VALIDATION_ROOT}/subsampled_plans.jsonl}"
REQUESTED_PLANS_PATH="${PLANS_PATH:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${VALIDATION_ROOT}/renders}"
SEPARATION_CACHE_DIR="${SEPARATION_CACHE_DIR:-${VALIDATION_ROOT}/demucs_cache}"
NUM_TRACKS="${NUM_TRACKS:-10}"
SEED="${SEED:-0}"
CANDIDATE_SCAN_LIMIT="${CANDIDATE_SCAN_LIMIT:-1500}"
MAX_VARIANTS_PER_RECIPE="${MAX_VARIANTS_PER_RECIPE:-}"
RANDOM_PLANS_PER_CLIP="${RANDOM_PLANS_PER_CLIP:-8}"
SUBSAMPLE_LIMIT="${SUBSAMPLE_LIMIT:-200}"
SUBSAMPLE_POLICY="${SUBSAMPLE_POLICY:-stratified_random}"
MAX_PER_SOURCE_RECIPE="${MAX_PER_SOURCE_RECIPE:-3}"
SUBSAMPLE_DEVICE="${SUBSAMPLE_DEVICE:-}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-}"

if [[ ! -f "${ALL_PLANS_PATH}" || "${REBUILD_VALIDATION_SUBSET:-0}" == "1" ]]; then
  prepare_args=(
    "${REPO_ROOT}/scripts/prepare_mtg_jamendo_validation_subset.py"
    --output-root "${VALIDATION_ROOT}"
    --num-tracks "${NUM_TRACKS}"
    --seed "${SEED}"
    --candidate-scan-limit "${CANDIDATE_SCAN_LIMIT}"
    --dataset-config "${REPO_ROOT}/configs/ground_truth/datasets/mtg_jamendo.yaml"
    --config-dir "${REPO_ROOT}/configs/ground_truth"
  )
  if [[ -n "${MAX_VARIANTS_PER_RECIPE}" ]]; then
    prepare_args+=(--max-variants-per-recipe "${MAX_VARIANTS_PER_RECIPE}")
  fi
  prepare_args+=(--random-plans-per-clip "${RANDOM_PLANS_PER_CLIP}")
  "${PYTHON_BIN}" "${prepare_args[@]}"
fi

if [[ "${SKIP_SUBSAMPLE:-0}" != "1" && ( ! -f "${SUBSAMPLED_PLANS_PATH}" || "${REBUILD_PLAN_SUBSAMPLE:-0}" == "1" || "${REBUILD_VALIDATION_SUBSET:-0}" == "1" ) ]]; then
  subsample_args=(
    "${REPO_ROOT}/scripts/subsample_ground_truth_plans.py"
    --plans-path "${ALL_PLANS_PATH}"
    --output-path "${SUBSAMPLED_PLANS_PATH}"
    --policy "${SUBSAMPLE_POLICY}"
    --limit "${SUBSAMPLE_LIMIT}"
    --max-per-source-recipe "${MAX_PER_SOURCE_RECIPE}"
    --seed "${SEED}"
  )
  if [[ -n "${SUBSAMPLE_DEVICE}" ]]; then
    subsample_args+=(--device "${SUBSAMPLE_DEVICE}")
  fi
  "${PYTHON_BIN}" "${subsample_args[@]}"
fi

if [[ -n "${REQUESTED_PLANS_PATH}" ]]; then
  PLANS_PATH="${REQUESTED_PLANS_PATH}"
elif [[ -f "${SUBSAMPLED_PLANS_PATH}" ]]; then
  PLANS_PATH="${SUBSAMPLED_PLANS_PATH}"
else
  PLANS_PATH="${ALL_PLANS_PATH}"
fi

render_args=(
  "${REPO_ROOT}/scripts/render_permissible_plans.py"
  --plans-path "${PLANS_PATH}"
  --output-root "${OUTPUT_ROOT}"
  --config-dir "${REPO_ROOT}/configs/ground_truth"
  --separation-cache-dir "${SEPARATION_CACHE_DIR}"
)
if [[ -n "${MAX_AUDIO_SECONDS}" ]]; then
  render_args+=(--max-audio-seconds "${MAX_AUDIO_SECONDS}")
fi
if [[ "${SKIP_HARMONY_RENDER:-0}" == "1" ]]; then
  render_args+=(--skip-harmony)
fi

exec "${PYTHON_BIN}" "${render_args[@]}" "$@"
