# 0001. Hand-written agent loop instead of LangGraph

- Status: accepted
- Date: 2026-11-07

## Context

The agent loop is the core of the project: it calls the model, validates tool
calls, asks the permission engine, runs tools in the sandbox, feeds results
(including errors) back as observations, and stops on turn, token, cost, time
and loop-detection budgets. Every one of those steps also has to emit a trace
span and be testable deterministically with a scripted fake model.

LangGraph and LangChain provide graphs, state handling, tool abstractions and
integrations. They would shorten the first demo, but:

- the loop is the main thing this project is meant to teach and the main
  interview topic, and a framework hides exactly that part;
- each framework layer sits between the code and the OpenAI-compatible request,
  which makes it harder to control the exact prompt prefix (needed for prefix
  caching in phase 5), the exact token accounting and the trace attributes;
- framework APIs change often, which works against pinned, reproducible evals;
- mini-swe-agent shows that a usable coding-agent loop fits in roughly a
  hundred lines, so the cost of writing it is small.

## Decision

Write the agent loop by hand in `src/agent/core/`, talking to the model
through the `openai` SDK against any OpenAI-compatible endpoint. No LangChain
or LangGraph.

## Consequences

- Full control over message layout, prompt prefix stability, retries, budgets
  and tracing; every behaviour is visible in project code.
- The project owns things a framework would provide: retry and timeout
  policy, conversation state, streaming, session persistence, subagents.
- Tests use a `FakeLLM` with scripted responses, so the loop is tested in CI
  without any model.
- Revisit if the loop grows features that a framework solves well and that
  are not themselves the point of the project (for example durable multi-step
  workflows with checkpoints across processes).
