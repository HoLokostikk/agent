# agent

A terminal coding agent built on an open-weight model (Qwen3.8-27B), with a
hand-written agent loop, sandboxed tools, evals with CI gates and tracing.

**Status:** phase 0 (scaffolding). No results yet — every number that appears
here later will link to the report it came from.

## Development

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

    uv sync
    make check    # lint + mypy --strict + tests

## License

Apache-2.0, see [LICENSE](LICENSE).
