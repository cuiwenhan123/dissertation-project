#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

export ROBUSTNESS_USE_REAL_MODELS="${ROBUSTNESS_USE_REAL_MODELS:-1}"
export ROBUSTNESS_ALLOW_MODEL_DOWNLOAD="${ROBUSTNESS_ALLOW_MODEL_DOWNLOAD:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
exec "${PYTHON:-python3}" server.py
