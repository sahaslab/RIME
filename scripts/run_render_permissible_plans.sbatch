#!/usr/bin/env bash
#SBATCH --job-name=pm-render
#SBATCH --account=sahaslab-gpu
#SBATCH --partition=l40s_nova
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT=/dartfs/rc/lab/S/SinghN/projects/RIME
cd "${REPO_ROOT}"

PLANS_PATH="${PLANS_PATH:-$HOME/lab/postmaster/ground_truth/retagged_subsampled_plans_100.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/lab/postmaster/retagged-generated-audio}"
CONFIG_DIR="${CONFIG_DIR:-$REPO_ROOT/configs/ground_truth}"
RENDER_SEED="${RENDER_SEED:-0}"
LIMIT_ARG=()
FAIL_FAST_ARG=()
SHUFFLE_ARG=()

if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "${LIMIT}")
fi
if [[ "${FAIL_FAST:-0}" == "1" ]]; then
  FAIL_FAST_ARG=(--fail-fast)
fi
if [[ "${DISABLE_SHUFFLE:-0}" == "1" ]]; then
  SHUFFLE_ARG=(--disable-shuffle)
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
MANIFEST_DIR="${OUTPUT_ROOT}/manifests"
mkdir -p "${MANIFEST_DIR}"
MANIFEST_PATH="${MANIFEST_PATH:-${MANIFEST_DIR}/render_manifest_${RUN_STAMP}.jsonl}"

if [[ -z "${CONDA_EXE:-}" ]]; then
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
  else
    echo "Could not find conda.sh; set up conda initialization for batch jobs." >&2
    exit 1
  fi
else
  source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
fi

conda activate postmaster-clean

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

printf 'repo=%s\nplans=%s\noutput_root=%s\nmanifest=%s\nseed=%s\nbackend=%s\ndemucs_model=%s\n' \
  "$REPO_ROOT" "$PLANS_PATH" "$OUTPUT_ROOT" "$MANIFEST_PATH" "$RENDER_SEED" \
  "${SEPARATION_BACKEND:-demucs}" "${DEMUCS_MODEL:-htdemucs_6s}"

bash "$REPO_ROOT/scripts/render_permissible_plans.sh" \
  --plans-path "$PLANS_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --manifest-path "$MANIFEST_PATH" \
  --config-dir "$CONFIG_DIR" \
  --seed "$RENDER_SEED" \
  "${LIMIT_ARG[@]}" \
  "${FAIL_FAST_ARG[@]}" \
  "${SHUFFLE_ARG[@]}"
