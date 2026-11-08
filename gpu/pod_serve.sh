#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_DIR="/workspace/models/$(basename "$(cat gpu/PINNED_MODEL)")"
args=(serve "$MODEL_DIR"
  --served-model-name "${SERVED_NAME:-qwen27b}"
  --host 127.0.0.1 --port 8000
  --max-model-len "${MAX_LEN:-32768}"
  --enable-auto-tool-choice --tool-call-parser "${TOOL_PARSER:-hermes}"
  --reasoning-parser "${REASONING_PARSER:-qwen3}"
  --gpu-memory-utilization "${GPU_UTIL:-0.92}")

if [ "${PREFIX_CACHING:-1}" = "0" ]; then
  args+=(--no-enable-prefix-caching)
else
  args+=(--enable-prefix-caching)
fi
if [ -n "${LORA:-}" ]; then
  # shellcheck disable=SC2206
  args+=(--enable-lora --max-lora-rank "${MAX_LORA_RANK:-64}" --lora-modules ${LORA})
fi
if [ -n "${QUANTIZATION:-}" ]; then
  args+=(--quantization "$QUANTIZATION")
fi

echo "vllm ${args[*]}"
exec vllm "${args[@]}"
