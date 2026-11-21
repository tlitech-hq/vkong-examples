#!/usr/bin/env bash
# One vLLM engine. Each machine of the App runs this script; VKong's URL spreads
# requests over the machines and keeps requests with the same X-Vkong-Session
# header on one machine, so that machine's prefix cache is reused.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
: "${LLM_API_KEY:?add LLM_API_KEY to the llm-session-routing secret (README step 2)}"

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"

# Metrics leave the machine by push: the public URL reaches the App, not one
# machine, so nothing outside can scrape each engine by itself.
if [ -n "${METRICS_WRITE_URL:-}" ]; then
  python3 metrics_agent.py &
else
  echo "METRICS_WRITE_URL is not set: serving without metrics push"
fi

exec vllm serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --api-key "$LLM_API_KEY" \
  --served-model-name "$MODEL" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details
