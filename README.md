# agent

A terminal coding agent built on an open-weight model (Qwen3.8-27B), with a
hand-written agent loop, sandboxed tools, evals with CI gates and tracing.

**Status:** phase 2 (agent core MVP). No results yet — every number that appears
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

## License

Apache-2.0, see [LICENSE](LICENSE).
