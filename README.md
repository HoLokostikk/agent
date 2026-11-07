# agent

A terminal coding agent built on an open-weight model (Qwen3.8-27B), with a
hand-written agent loop, sandboxed tools, evals with CI gates and tracing.

**Status:** phase 1 (model access layer). No results yet — every number that appears
here later will link to the report it came from.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

    uv sync
    make check      # lint + mypy --strict + tests
    make test-live  # tests against a running local Ollama

## Usage

    agent models                                  # model registry
    agent chat -m local-small "Explain a hash map"  # one-shot
    agent chat -m local-small                     # REPL, /exit to quit

Every call prints tokens, cost, latency and TTFT. Hosted models read their
API key from `.env` (see `.env.example`).

## License

Apache-2.0, see [LICENSE](LICENSE).
