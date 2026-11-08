#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-training/configs/qwen27b-qlora.yaml}"
RUN="${RUN:-qwen27b-r16}"
mkdir -p training/runs
python3 gpu/make_run_config.py "$CONFIG" "training/runs/${RUN}.yaml" "$RUN"
python3 training/sft.py --config "training/runs/${RUN}.yaml" --show-masking 1 | head -80
python3 training/sft.py --config "training/runs/${RUN}.yaml" 2>&1 | tee "training/runs/${RUN}.log"
