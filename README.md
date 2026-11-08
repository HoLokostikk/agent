# agent

A terminal coding agent on an open-weight model (Qwen3.8-27B), built to be
measured: hand-written agent loop, Docker-sandboxed tools, permissions with a
human in the loop, code RAG, MCP, evals with CI gates, OpenTelemetry tracing,
a QLoRA distillation pipeline and vLLM serving.

**Status:** all phases implemented; the numbers that need the 27B model are
pending a GPU run (`gpu/README.md`). Every number below links to the report it
came from; a dash means "not measured yet".

## Results

| what | value | source |
|---|---|---|
| golden set pass@1, 27B base | — | `gpu/laptop_measure.sh baseline` |
| golden set pass@1, 27B + QLoRA | — | `gpu/laptop_measure.sh sft-eval` |
| tool-call accuracy, 3B local (30 cases) | see `EXPERIMENTS.md`, phase 3 | `evals/results/` |
| retrieval recall@10, BM25 vs hybrid | — | `agent eval retrieval` |
| TTFT with / without prefix caching | — | `gpu/laptop_measure.sh bench` |
| sandbox security properties | 7 checks, in CI | `tests/sandbox/test_docker.py` |

## Architecture

    you -> CLI (REPL | headless -p) -> agent loop (src/agent/core)
                                         |-- tools: read/list/glob/grep/edit/write/bash, search_code, todo, explore
                                         |     '-- bash -> Docker sandbox (no network, no caps, read-only root)
                                         |-- permissions: read-only | ask | auto, deny rules, approval with diff
                                         |-- context: AGENTS.md, compaction, sessions (--resume)
                                         |-- MCP client -> MCP servers (incl. our code-search server)
                                         '-- OpenTelemetry spans (redacted) -> JSONL / OTLP
                                       -> OpenAI-compatible endpoint: Ollama | vLLM | LiteLLM gateway | hosted API

## Quickstart

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

    uv sync
    make sandbox
    agent models                               # model registry
    agent                                      # REPL in the current repo (ask mode)
    agent -p "fix the failing test" --output json
    agent --rag --embed-model local-embed      # with semantic code search
    agent --mcp                                # with the MCP servers in configs/mcp.json

Hosted models read their API key from `.env` (see `.env.example`).

## Evals

    agent eval toolcalls -m <alias>            # 30 single-step tool-call cases
    agent eval tasks -m <alias> -k 3           # golden set, pass@1 / pass^3 with bootstrap CIs
    agent eval retrieval --repo <git repo>     # BM25 / vector / hybrid on git history
    agent eval compare BASE.json CAND.json     # regression gate (paired bootstrap)

Golden tasks (`evals/tasks/`) are checked to fail as shipped and pass with
their reference solution; tests are restored before grading, so editing them
never counts as a fix. Evals refuse to start on a dead endpoint or an unpinned
model revision.

## Training and serving

    agent data mutate --repo <python repo> --out training/tasks   # verified bug-fix tasks
    agent data collect --tasks-dir training/tasks -m <teacher>    # rejection-sampled SFT data
    python training/sft.py --config training/configs/qwen27b-qlora.yaml
    agent bench -m <alias> --concurrency 1,8,32                   # TTFT, ITL, throughput

The full GPU procedure, costs and reproducibility checklist: `gpu/README.md`.
Release and rollback: `docs/RUNBOOK.md`. Threat model: `docs/SECURITY.md`.

## Development

    make check        # ruff + mypy --strict + tests
    make test-docker  # sandbox and golden-set oracle tests against real containers
    make test-live    # against a local Ollama

## Docs

`docs/design.md` (goals, metrics) · `docs/status.md` (state, open items) ·
`EXPERIMENTS.md` (journal) · `docs/failure_taxonomy.md` · `docs/adr/` ·
`docs/MODEL_CARD.md` · `docs/DATASET_CARD.md`

## License

Apache-2.0, see [LICENSE](LICENSE).
