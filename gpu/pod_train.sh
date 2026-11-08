#!/usr/bin/env bash
# Run ON THE GPU POD after training/data/out/ was copied over from the laptop.
#
#   bash gpu/pod_train.sh                                    # main run, r=16
#   CONFIG=training/configs/rehearsal-small.yaml RUN=rehearsal bash gpu/pod_train.sh
#   LORA_R=64 RUN=qwen27b-r64 bash gpu/pod_train.sh          # rank ablation
#   FRACTION=0.25 RUN=qwen27b-25pct bash gpu/pod_train.sh    # data-size ablation
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${CONFIG:-training/configs/qwen27b-qlora.yaml}"
RUN="${RUN:-qwen27b-r16}"
mkdir -p training/runs
python3 gpu/make_run_config.py "$CONFIG" "training/runs/${RUN}.yaml" "$RUN"
python3 training/sft.py --config "training/runs/${RUN}.yaml" --show-masking 1 | head -80
python3 training/sft.py --config "training/runs/${RUN}.yaml" 2>&1 | tee "training/runs/${RUN}.log"
