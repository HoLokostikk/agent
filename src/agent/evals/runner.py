"""Run the agent over the golden set: k runs per task, each in a fresh copy.

Grading happens in a second, fresh sandbox after the agent is done, with
protected files (the tests) restored from the snapshot and hidden tests
overlaid, so neither a leftover process nor an edited test can fake a pass.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from agent.core import Budget, ChatModel, RunResult
from agent.core.builder import AgentSpec, build_agent
from agent.evals.stats import (
    Outcomes,
    bootstrap_ci,
    mean_pass_at_k,
    mean_pass_hat_k,
)
from agent.evals.tasks import Task, prepare_grading, prepare_workspace
from agent.permissions import Mode, Rules
from agent.sandbox import DockerSandbox, Sandbox, SandboxError
from agent.tools.truncate import head_tail

SandboxFactory = Callable[[Path], AbstractAsyncContextManager[Sandbox]]
Progress = Callable[["RunRecord"], None]


@asynccontextmanager
async def docker_sandbox(root: Path) -> AsyncIterator[Sandbox]:
    async with DockerSandbox(root) as box:
        yield box


@dataclass(slots=True)
class RunRecord:
    task_id: str
    attempt: int
    passed: bool
    stop_reason: str | None
    tampered: list[str]
    grade_exit_code: int | None
    grade_output: str
    stats: dict[str, Any]
    turns: list[dict[str, Any]]
    wall_s: float
    error: str | None = None  # infrastructure failure, not an agent failure
    trace_id: str | None = None
    messages: list[dict[str, Any]] | None = None  # kept for training data only
    tools: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class SuiteConfig:
    model: ChatModel
    rules: Rules
    system_prompt: str
    prompt_version: str
    k: int = 3
    concurrency: int = 1
    budget: Budget = field(default_factory=Budget)
    sandbox_factory: SandboxFactory = docker_sandbox
    compaction: bool = True
    compact_model: ChatModel | None = None
    keep_messages: bool = False


async def run_once(task: Task, attempt: int, cfg: SuiteConfig, workdir: Path) -> RunRecord:
    started = time.monotonic()
    ws = prepare_workspace(task, workdir / f"{task.id}-{attempt}")
    result: RunResult | None = None
    try:
        async with cfg.sandbox_factory(ws) as sandbox:
            spec = AgentSpec(
                base_prompt=cfg.system_prompt,
                prompt_version=cfg.prompt_version,
                rules=cfg.rules,
                mode=Mode.AUTO,
                budget=replace(cfg.budget, max_turns=min(cfg.budget.max_turns, task.max_turns)),
                compaction=cfg.compaction,
            )
            agent = build_agent(
                cfg.model,
                ws,
                sandbox,
                spec,
                compact_model=cfg.compact_model,
                trace_attributes={
                    "agent.session": "eval",
                    "agent.eval.task_id": task.id,
                    "agent.eval.attempt": attempt,
                },
            )
            result = await agent.run(task.instruction)
        tampered = prepare_grading(task, ws)
        async with cfg.sandbox_factory(ws) as grader:
            graded = await grader.exec(task.test_command, task.test_timeout_s)
    except (SandboxError, OSError) as exc:
        return RunRecord(
            task_id=task.id,
            attempt=attempt,
            passed=False,
            stop_reason=result.stop_reason.value if result else None,
            tampered=[],
            grade_exit_code=None,
            grade_output="",
            stats=result.stats.as_dict() if result else {},
            turns=[asdict(t) for t in result.turns] if result else [],
            wall_s=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    return RunRecord(
        task_id=task.id,
        attempt=attempt,
        passed=graded.exit_code == 0 and not graded.timed_out,
        stop_reason=result.stop_reason.value,
        tampered=tampered,
        grade_exit_code=graded.exit_code,
        grade_output=head_tail(graded.output, max_lines=40, max_chars=3000),
        stats=result.stats.as_dict(),
        turns=[asdict(t) for t in result.turns],
        wall_s=time.monotonic() - started,
        trace_id=result.trace_id,
        messages=[dict(m) for m in agent.messages] if cfg.keep_messages else None,
        tools=[dict(t) for t in agent.tools.schemas()] if cfg.keep_messages else None,
    )


async def run_suite(
    tasks: Sequence[Task], cfg: SuiteConfig, *, progress: Progress | None = None
) -> list[RunRecord]:
    semaphore = asyncio.Semaphore(cfg.concurrency)
    with tempfile.TemporaryDirectory(prefix="agent-eval-") as tmp:
        workdir = Path(tmp)

        async def one(task: Task, attempt: int) -> RunRecord:
            async with semaphore:
                record = await run_once(task, attempt, cfg, workdir)
            if progress is not None:
                progress(record)
            return record

        jobs = [one(task, attempt) for task in tasks for attempt in range(1, cfg.k + 1)]
        records = await asyncio.gather(*jobs)
    return sorted(records, key=lambda r: (r.task_id, r.attempt))


def _per_tool_error_rates(records: Sequence[RunRecord]) -> dict[str, dict[str, float]]:
    calls: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for record in records:
        for turn in record.turns:
            for call in turn["tool_calls"]:
                calls[call["name"]] += 1
                if not call["ok"]:
                    errors[call["name"]] += 1
    return {
        name: {
            "calls": calls[name],
            "errors": errors[name],
            "error_rate": errors[name] / calls[name],
        }
        for name in sorted(calls)
    }


def summarize(records: Sequence[RunRecord], k: int) -> dict[str, Any]:
    if not records:
        raise ValueError("no runs to summarize")
    outcomes: dict[str, list[bool]] = {}
    for record in records:
        outcomes.setdefault(record.task_id, []).append(record.passed)
    n_min = min(len(v) for v in outcomes.values())

    def pass1(o: Outcomes) -> float:
        return mean_pass_at_k(o, 1)

    def passk(o: Outcomes) -> float:
        return mean_pass_hat_k(o, n_min)

    p1 = bootstrap_ci(outcomes, pass1)
    pk = bootstrap_ci(outcomes, passk)
    agent_runs = [r for r in records if r.stats]
    total_calls = sum(r.stats.get("tool_calls", 0) for r in agent_runs)
    invalid = sum(r.stats.get("invalid_tool_calls", 0) for r in agent_runs)
    cost_known = all(r.stats.get("cost_known", False) for r in agent_runs)
    total_cost = sum(r.stats.get("cost_usd", 0.0) for r in agent_runs)
    passed = sum(r.passed for r in records)

    def mean(key: str) -> float:
        return sum(r.stats.get(key, 0) for r in agent_runs) / len(agent_runs) if agent_runs else 0.0

    return {
        "tasks": len(outcomes),
        "runs": len(records),
        "k": n_min,
        "pass_at_1": asdict(p1),
        f"pass_hat_{n_min}": asdict(pk),
        f"pass_at_{n_min}": mean_pass_at_k(outcomes, n_min),
        "passed_runs": passed,
        "mean_turns": mean("turns"),
        "mean_input_tokens": mean("input_tokens"),
        "mean_output_tokens": mean("output_tokens"),
        "mean_wall_s": sum(r.wall_s for r in records) / len(records),
        "total_cost_usd": total_cost if cost_known else None,
        "cost_per_task_usd": total_cost / len(records) if cost_known else None,
        "cost_per_successful_task_usd": (total_cost / passed) if cost_known and passed else None,
        "invalid_tool_call_rate": invalid / total_calls if total_calls else None,
        "tool_calls": total_calls,
        "tool_error_rates": _per_tool_error_rates(records),
        "stop_reasons": dict(Counter(r.stop_reason or "none" for r in records)),
        "tampered_runs": sum(bool(r.tampered) for r in records),
        "infra_errors": sum(r.error is not None for r in records),
        # A run that never reached the model measures the endpoint, not the agent.
        "model_errors": sum(r.stop_reason == "model_error" for r in records),
        "per_task": {
            task: {"passed": sum(runs), "runs": len(runs)}
            for task, runs in sorted(outcomes.items())
        },
    }


def git_revision(repo: Path) -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha


def results_payload(
    kind: str,
    model: ChatModel,
    prompt_version: str,
    summary: dict[str, Any],
    runs: list[dict[str, Any]],
    extra: dict[str, Any] | None = None,
    git: str | None = None,
) -> dict[str, Any]:
    """``git`` should be captured before the runs start: editing code during a
    long eval would otherwise label the results with the wrong revision."""
    cfg = model.config
    return {
        "schema_version": 1,
        "kind": kind,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": git or git_revision(Path.cwd()),
        "model": {
            "alias": cfg.alias,
            "model": cfg.model,
            "revision": cfg.revision,
            "sampling": cfg.sampling.model_dump(exclude_none=True),
        },
        "prompt_version": prompt_version,
        **(extra or {}),
        "summary": summary,
        "runs": runs,
    }


def results_path(out_dir: Path, kind: str, alias: str, prompt_version: str) -> Path:
    stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H%M")
    return out_dir / f"{stamp}_{kind}_{alias}_{prompt_version}.json"
