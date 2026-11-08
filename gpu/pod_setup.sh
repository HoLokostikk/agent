#!/usr/bin/env bash
# Run ON THE GPU POD, once. Installs vLLM and training deps, resolves the exact
# model revision and downloads the weights at that revision.
#
#   MODEL_ID=Qwen/Qwen3.8-27B bash gpu/pod_setup.sh
#
# Writes gpu/PINNED_REVISION: every later step reads it, so baseline, data
# generation, training and the fine-tuned eval all use the same weights.
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
VLLM_VERSION="${VLLM_VERSION:-}"   # empty: latest; the installed version is recorded below
cd "$(dirname "$0")/.."

python3 -m pip install --upgrade pip
if [ -n "$VLLM_VERSION" ]; then
  python3 -m pip install "vllm==${VLLM_VERSION}"
else
  python3 -m pip install vllm
fi
python3 -m pip install -r training/requirements.txt huggingface_hub
python3 -m pip install -e .

if [ -f gpu/PINNED_REVISION ]; then
  REVISION="$(cat gpu/PINNED_REVISION)"
  echo "using pinned revision ${REVISION}"
else
  REVISION="$(python3 gpu/resolve_revision.py "$MODEL_ID")"
  echo "$REVISION" > gpu/PINNED_REVISION
  echo "pinned ${MODEL_ID} at ${REVISION}"
fi
echo "$MODEL_ID" > gpu/PINNED_MODEL

huggingface-cli download "$MODEL_ID" --revision "$REVISION" \
  --local-dir "/workspace/models/$(basename "$MODEL_ID")"
sed -i "s/^revision: UNPINNED.*/revision: ${REVISION}/" training/configs/qwen27b-qlora.yaml

python3 -m pip freeze > gpu/pod_freeze.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > gpu/pod_gpu.txt
echo "done: $(tail -1 gpu/pod_gpu.txt)"
