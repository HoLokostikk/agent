# agent — terminal coding agent on an open-weight model

Portfolio project. A CLI coding agent (in the spirit of Claude Code / Aider /
OpenHands) built on **Qwen3.8-27B**, with its own agent loop, sandboxed tools,
code RAG, MCP, evals with CI gates, tracing, QLoRA distillation and vLLM serving.

The point of the project is **measured numbers**, not a feature clone. Every
claim in the README must trace back to a report or a run.

Full plan: `~/qwen-agent-plan.md` (kept outside the repo: personal, in Ukrainian).
Current state and decisions: `docs/status.md`.

## Working agreement

- **Communicate in Ukrainian.** Code, comments, docstrings, commit messages and
  all repo documentation in English.
- **Claude writes all the code** (decided 2026-11-07, reversing the earlier
  split). Work autonomously, phase by phase, without stopping after each step.
  Stop only for what needs the author: money, credentials, accounts, or a
  decision that changes the plan's goals.
- **Scope.** Fixes needed to finish the current phase are in scope; note them
  in the journal. Anything that changes the plan goes into `docs/status.md`.
- **Every phase ends in a working state:** `make check` green, a commit, and an
  entry in `EXPERIMENTS.md`.
- **Evals come before optimisation and before fine-tuning.** A number without a
  baseline means nothing.

## Journal format (`EXPERIMENTS.md`, Ukrainian)

Question of the day → options considered → what was done → numbers →
conclusion → what is still unclear.

The "options considered" line is written *before* implementing, not after: it
forces naming the alternative while it is still a choice.

## Stack

Python 3.12 · uv · ruff · mypy --strict · pytest · Typer + Rich ·
openai SDK (any OpenAI-compatible endpoint) · Docker sandbox · OpenTelemetry

Agent loop is hand-written. **No LangChain or LangGraph** — the loop is the
thing being learned here, and it is the main interview topic.

## Commands

    make fmt      format and autofix
    make check    lint + mypy + tests  (must be green before every commit)
    make test     tests only
    make eval     golden set + tool-call eval on local-small (slow)

## Layout

    src/agent/
      cli/          REPL, headless mode, slash commands
      llm/          client, retries, token and cost accounting, routing
      core/         agent loop, context manager, compaction, subagents
      tools/        read/edit/write/grep/glob/bash + schemas
      sandbox/      docker runner
      permissions/  policy engine, human-in-the-loop
      rag/          indexer, chunker, retrievers, reranker
      mcp/          client + code-search server
      telemetry/    OTel setup, redaction
      evals/        runner, statistics, tool-call eval, reports, gate
    evals/          data: tasks, tool_calls, baselines.yaml, results
    training/       data generation and fine-tuning
    serving/        vLLM, LiteLLM

`core/` holds the agent loop — named `core` and not `agent` to avoid
`agent.agent` import paths.

## Rules that are easy to get wrong

- Never put secrets in prompts, logs or traces. `.env` is gitignored.
- The model's output is never executed outside the Docker sandbox.
- A tool error is returned to the model as an observation, never raised out of
  the loop.
- Pin model revisions. A moving alias makes every eval number unreproducible.
