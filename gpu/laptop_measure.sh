#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

step="${1:?step: baseline | bench | bench-nocache | tasks | collect | sft-eval | compare}"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a gpu/measure.log; }

pin_revision() {
  local rev
  if [ -n "${POD_SSH:-}" ]; then
    # shellcheck disable=SC2086
    rev="$(ssh ${POD_SSH} cat /workspace/agent/gpu/PINNED_REVISION)"
  else
    rev="$(cat gpu/PINNED_REVISION)"
  fi
  [ -n "$rev" ] || { echo "no pinned revision"; exit 1; }
  sed -i "s|revision: UNPINNED-qwen27b-vllm|revision: hf:${rev}|" configs/models.yaml
  grep -q "hf:${rev}" configs/models.yaml || { echo "models.yaml revision != pod revision ${rev}"; exit 1; }
}

case "$step" in
  baseline)
    pin_revision
    for i in 1 2 3; do uv run agent eval toolcalls -m qwen27b-vllm; done
    uv run agent eval tasks -m qwen27b-vllm -k 3 --concurrency 4
    log "baseline done"
    ;;
  bench)
    uv run agent bench -m qwen27b-vllm --concurrency 1,8,32 --label "cache on"
    uv run agent bench -m qwen27b-vllm --concurrency 1,8,32 --shared-prefix-chars 6000 --label "shared prefix cache on"
    log "bench (cache on) done"
    ;;
  bench-nocache)
    uv run agent bench -m qwen27b-vllm --concurrency 1,8,32 --shared-prefix-chars 6000 --label "shared prefix cache off"
    log "bench (cache off) done"
    ;;
  tasks)
    mkdir -p training/sources training/tasks
    for repo in pallets/click pallets/itsdangerous python-attrs/attrs pallets/markupsafe; do
      name="$(basename "$repo")"
      [ -d "training/sources/$name" ] || git clone -q "https://github.com/$repo" "training/sources/$name"
    done
    make sandbox-train
    for name in click itsdangerous attrs markupsafe; do
      uv run agent data mutate --repo "training/sources/$name" --out training/tasks -n "${PER_REPO:-150}" \
        --image agent-sandbox-train:0.1 || log "mutate $name failed"
    done
    log "tasks: $(ls training/tasks | wc -l)"
    ;;
  collect)
    pin_revision
    uv run agent data collect --tasks-dir training/tasks -m qwen27b-vllm -k 2 --concurrency 4
    log "collect: $(tr -d '\n' < training/data/out/stats.json | head -c 400)"
    ;;
  sft-eval)
    for i in 1 2 3; do uv run agent eval toolcalls -m qwen27b-ft-v1; done
    uv run agent eval tasks -m qwen27b-ft-v1 -k 3 --concurrency 4
    log "sft eval done"
    ;;
  compare)
    base="$(ls -t evals/results/*_tasks_qwen27b-vllm_*.json | head -1)"
    cand="$(ls -t evals/results/*_tasks_qwen27b-ft-v1_*.json | head -1)"
    uv run agent eval compare "$base" "$cand" --max-drop 1.0 | tee -a gpu/measure.log
    ;;
  *) echo "unknown step: $step"; exit 2 ;;
esac
