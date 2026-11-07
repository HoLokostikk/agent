# Design

Status: accepted, 2026-11-07. Thresholds below are set **before** any numbers
exist; changing one later needs a written reason in `EXPERIMENTS.md`.

## Problem

Terminal coding agents (Claude Code, Aider, OpenHands, Qwen Code,
mini-swe-agent) already exist, so another CLI is not the contribution. What is
rarely shown end to end, with numbers, is the full engineering cycle around
such an agent on an **open-weight model you can host yourself**:

harness → evals → test-verified training data → fine-tune → serving →
monitoring.

Most write-ups stop at a demo. Without a baseline and confidence intervals it
is impossible to say whether a prompt change, a retrieval layer or a
fine-tune actually helped. This project builds that cycle on Qwen3.8-27B and
reports every claim against a measured baseline.

## Goals

1. **A working agent.** A hand-written tool-use loop that, in headless mode,
   finds and fixes a failing test in a small repository, with every tool call
   executed inside a Docker sandbox.
2. **Model-agnostic harness.** The same code runs against any
   OpenAI-compatible endpoint (Ollama, vLLM, hosted API); switching models is
   a config change.
3. **Evals that gate changes.** A golden task set and a tool-call suite,
   reported as pass@1 and pass^3 with 95% bootstrap CIs, wired into CI as a
   regression gate.
4. **A measured fine-tune.** A QLoRA adapter trained on test-verified
   trajectories from an open-weight teacher, compared to the base model in the
   same harness with the same prompts.
5. **Production concerns, measured.** Tracing, cost per successful task,
   prompt-injection attack success rate, and a timed model rollback.

## Non-goals

- Not an IDE plugin and not a multi-user SaaS.
- Not feature parity with Claude Code or any other product.
- Not a new state of the art on SWE-bench. The external benchmark is a
  sanity check, not the target.
- No agent frameworks (LangChain, LangGraph): the loop is the thing being
  built, see [ADR 0001](adr/0001-own-agent-loop.md).
- No training on outputs of Claude, GPT or any model whose terms forbid it.
- No GPU work on the development laptop; heavy jobs go to rented or free
  cloud GPUs.

## Success metrics

| # | metric | threshold | measured in |
|---|---|---|---|
| 1 | golden set size | ≥ 30 tasks, each with hidden tests | `evals/tasks/` |
| 2 | baseline reported | pass@1 and pass^3 with 95% bootstrap CI for the small local model and for 27B base | `evals/results/` (phase 3) |
| 3 | regression gate | CI fails a PR whose smoke pass@1 drops by more than a threshold derived from run-to-run variance; the derivation is written down | `.github/workflows/`, `EXPERIMENTS.md` |
| 4 | invalid tool call rate, 27B base | reported per tool; the fine-tune target is a reduction whose 95% CI excludes zero | phase 3 and 10 reports |
| 5 | fine-tune effect | SFT vs base pass@1 difference reported with 95% CI; a null or negative result is documented with analysis, not hidden | phase 10 report |
| 6 | retrieval | recall@5, recall@10 and MRR for BM25, vector, hybrid and hybrid + rerank | `evals/retrieval/` (phase 6) |
| 7 | prompt injection | ASR on ≥ 30 cases before and after mitigations; after ≤ 10% | `evals/injection/` (phase 8) |
| 8 | command classifier | false positive rate ≤ 5% on a benign command set | phase 8 report |
| 9 | rollback | model alias switched back in < 60 s in a timed drill | `docs/RUNBOOK.md` (phase 11) |
| 10 | cost | every eval report includes cost per successful task; phases 0–8 developed at $0 compute; total project ≤ $300 | eval reports, `EXPERIMENTS.md` |
| 11 | traceability | every number in the README links to the report or trace it came from | `README.md` |

## Risks

| risk | likelihood | impact | mitigation |
|---|---|---|---|
| **Compute.** 7 GB RAM, no discrete GPU; 27B cannot run locally | certain | high | Develop phases 0–8 against `qwen2.5:3b` in Ollama; use 27B only for phase 3 baselines, phase 9 generation and phase 10 training, on hosted API or rented GPU |
| **Thinking mode inflates context and cost.** 27B thinks by default (`xhigh`, `preserve_thinking`) | high | medium | Handle `reasoning_content` explicitly in the client; measure thinking on vs off on evals instead of guessing; budget phase 9 on a rented GPU, not the API |
| **Unverified model facts.** Context size, sampling defaults and revision come from third-party pages | medium | high | Read the official model card before phase 1; pin the model revision; never use a moving alias in evals |
| **Licensing and ToS.** Teacher output or a provider's terms may forbid training | medium | high | Teacher only from open-weight models whose licence allows distillation; check model licence and provider ToS; record both in `DATASET_CARD.md` (ADR 0003) |
| **Contamination.** Eval tasks may appear in pretraining or in training data | medium | high | Split train/val by repository; decontaminate train against all eval sets; prefer self-authored golden tasks |
| **Small samples.** 30–50 tasks give wide confidence intervals | certain | medium | Always report CIs; use pass^k and multiple runs; do not claim a difference whose CI includes zero |
| **Time.** ~200–250 h total | high | medium | Every phase ends in a working, committed state; minimal path (phases 0–4, 7, 9-lite, 10-lite, 12) defined in the plan |
| **Running model output on the host** | low | critical | Model output is executed only inside the Docker sandbox, `--network none`, non-root, path guard |
