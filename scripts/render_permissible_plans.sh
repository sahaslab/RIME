#!/usr/bin/env bash
set -euo pipefail

export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEFAULT_OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/lab/postmaster/retagged-generated-audio-full}"

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--plans-path PATH] [--output-root PATH] [--manifest-path PATH] [--config-dir PATH] [--separation-cache-dir PATH] [--limit N] [--workers N] [--overwrite] [--fail-fast]

This is a thin wrapper around scripts/render_permissible_plans.py.
It launches one backend MCP server per active worker and sends queue-backed
render jobs through the same long-lived worker architecture used by the agent
pipeline.
When --workers > 1, the Python script fans out clip batches across multiple
backend server processes and merges results into one ordered manifest.

Backend defaults:
  separation backend: demucs
  demucs model: htdemucs_6s
  server config: zero_shot_agent.toml

Optional env:
  PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  DEMUCS cache default: <output-root>/_demucs_cache

If you explicitly switch the backend to sam_audio in zero_shot_agent.toml, the
server still respects the SAM_AUDIO_* environment variables.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/scripts/render_permissible_plans.py" \
  --plans-path "$HOME/lab/postmaster/ground_truth/retagged_subsample.jsonl" \
  --output-root "${DEFAULT_OUTPUT_ROOT}" \
  --config-dir "${REPO_ROOT}/configs/ground_truth" \
  --workers 16
  "$@"
