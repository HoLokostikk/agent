# agent

A terminal coding agent built on an open-weight model (Qwen3.8-27B), with a
hand-written agent loop, sandboxed tools, evals with CI gates and tracing.

**Status:** phases 0–2 done; phase 3 (evals) and 4 (tracing) implemented, baselines pending. No results yet — every number that appears
here later will link to the report it came from.

## Development

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

    uv sync
    make sandbox      # build the sandbox image (once)
    make check        # lint + mypy --strict + tests
    make test-docker  # sandbox security tests against real containers
    make test-live    # tests against a running local Ollama

## Usage

    agent                                          # REPL in the current repo
    agent -p "fix the failing test" --output json  # one task, headless
    agent chat -m local-small "hello"              # plain chat, no tools
    agent models                                   # model registry

The agent reads and edits files inside the workspace only, and runs every
command in a Docker container with no network, no capabilities, a read-only
root filesystem and the workspace as its only mount. In the REPL the default
permission mode is `ask` (writes and commands need confirmation, with a diff);
headless runs default to `auto` because there is no one to ask. Rules for
denied commands and protected paths are in `configs/permissions.yaml`.

Every model call reports tokens, cost, latency and TTFT; every run reports its
stop reason, turns, tool calls and invalid tool calls. Hosted models read their
API key from `.env` (see `.env.example`).

## Tracing

Spans go to `~/.local/state/agent/traces.jsonl` by default (`--trace none |
file | console | otlp`; otlp follows `OTEL_EXPORTER_OTLP_ENDPOINT`, e.g. a local
Phoenix). Spans carry metrics and decisions, never message content, and every
attribute is redacted for keys, tokens and emails.

    agent trace list          # recent runs with trace ids
    agent trace show <id>     # run → turns → model and tool calls
    agent trace stats         # p50/p95 latency and TTFT, tool failure rates

## Evals

    agent eval toolcalls -m local-small        # 30 single-step tool-call cases
    agent eval tasks -m local-small -k 3       # golden set, 3 runs per task
    agent eval compare BASE.json CAND.json     # regression gate

Golden tasks live in `evals/tasks/<id>/` (`repo/` the agent sees, `hidden/`
extra tests, `solution/` the reference fix). Every task is checked to fail as
shipped and pass with its solution. Before grading, the tests are restored from
the snapshot, so editing them never counts as a fix. Reports give pass@1 and
pass^k with 95% bootstrap intervals over tasks; results land in
`evals/results/` as JSON plus a markdown report.

## License

Apache-2.0, see [LICENSE](LICENSE).
