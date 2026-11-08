#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
REPOS="${REPOS:-pallets/click psf/requests pallets/flask}"

python3 -m pip install -q "sentence-transformers>=3"
rev="$(python3 gpu/resolve_revision.py "$EMBED_MODEL")"
echo "$rev" > gpu/PINNED_EMBED_REVISION
grep -q "pod-embed:" configs/models.yaml || cat >> configs/models.yaml <<YAML

  pod-embed:
    kind: embedding
    base_url: http://127.0.0.1:8001/v1
    model: ${EMBED_MODEL}
    revision: hf:${rev}
    context_window: 512
    pricing: {input_per_mtok: 0.0, output_per_mtok: 0.0}
YAML

vllm serve "$EMBED_MODEL" --revision "$rev" --task embed --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.15 > gpu/embed_server.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do
  curl -s -o /dev/null http://127.0.0.1:8001/v1/models && break
  sleep 5
done

mkdir -p training/sources
for repo in $REPOS; do
  name="$(basename "$repo")"
  [ -d "training/sources/$name" ] || git clone -q "https://github.com/$repo" "training/sources/$name"
  agent eval retrieval --repo "training/sources/$name" --max-commits 300
  agent eval retrieval --repo "training/sources/$name" --max-commits 300 --embed-model pod-embed --rerank
done
ls -t evals/results/*_retrieval_* | head -6
