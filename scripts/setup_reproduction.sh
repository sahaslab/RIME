#!/usr/bin/env bash
set -euo pipefail
rime_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rime_env="${1:-$rime_root/.conda}"
conda create --yes --prefix "$rime_env" python=3.11 pip
"$rime_env/bin/python" -m pip install uv==0.6.14
"$rime_env/bin/uv" pip install --python "$rime_env/bin/python" -r "$rime_root/requirements-priors.txt"
