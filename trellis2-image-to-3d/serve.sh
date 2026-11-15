#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_DISABLE_XET=1
unset HF_HUB_ENABLE_HF_TRANSFER
export HF_HUB_DOWNLOAD_TIMEOUT=120
export GRADIO_ANALYTICS_ENABLED=False
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT="${PORT:-7860}"
exec python -u launch.py
