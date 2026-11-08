import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent.cli import app as cli
from agent.evals.report import gate, render
from agent.evals.runner import SuiteConfig, results_payload, run_suite, summarize
from agent.evals.tasks import load_tasks
from agent.permissions import load_rules
from agent.sandbox import ExecResult
from tests.fakes import FakeLLM, FakeSandbox, call, say, use

ROOT = Path(__file__).parents[2]
TASKS = load_tasks(ROOT / "evals" / "tasks", ids=["wrong-operator-add"])
RULES = load_rules(ROOT / "configs" / "permissions.yaml")


class GradingSandbox(FakeSandbox):
    """Runs the task's test command for real, but on the host and only in tests.

    Grading must see the files the agent wrote, so this fake decides the exit
    code by inspecting the workspace instead of executing anything.
    """

    async def exec(self, command: str, timeout_s: float) -> ExecResult:
        self.commands.append(command)
        fixed = "return a + b" in (self.root / "calc.py").read_text()
        return ExecResult(0 if fixed else 1, "3 passed" if fixed else "1 failed")


def factory() -> object:
    @contextlib.asynccontextmanager
    async def make(root: Path) -> AsyncIterator[GradingSandbox]:
        yield GradingSandbox(root)

    return make


def fixing_steps() -> list:  # type: ignore[type-arg]
    return [
        use(
            call("edit_file", path="calc.py", old_string="return a - b", new_string="return a + b")
        ),
        say("fixed"),
    ]


def cheating_steps() -> list:  # type: ignore[type-arg]
    return [
        use(call("write_file", path="test_calc.py", content="def test_ok() -> None:\n    pass\n")),
        say("tests pass now"),
    ]


async def test_runner_grades_each_attempt_in_a_fresh_copy() -> None:
    llm = FakeLLM([*fixing_steps(), say("I give up")])
    cfg = SuiteConfig(
        model=llm,
        rules=RULES,
        system_prompt="s",
        prompt_version="system.v1",
        k=2,
        sandbox_factory=factory(),  # type: ignore[arg-type]
    )

    records = await run_suite(TASKS, cfg)

    assert [(r.attempt, r.passed) for r in records] == [(1, True), (2, False)]
    assert "return a - b" in (TASKS[0].repo / "calc.py").read_text()
    summary = summarize(records, 2)
    assert summary["pass_at_1"]["estimate"] == 0.5
    assert summary["pass_hat_2"]["estimate"] == 0.0
    assert summary["pass_at_2"] == 1.0
    assert summary["per_task"] == {"wrong-operator-add": {"passed": 1, "runs": 2}}


async def test_editing_the_tests_does_not_count_as_a_fix() -> None:
    llm = FakeLLM(cheating_steps())
    cfg = SuiteConfig(
        model=llm,
        rules=RULES,
        system_prompt="s",
        prompt_version="v",
        k=1,
        sandbox_factory=factory(),  # type: ignore[arg-type]
    )

    records = await run_suite(TASKS, cfg)

    assert not records[0].passed
    assert records[0].tampered == ["test_calc.py"]
    assert summarize(records, 1)["tampered_runs"] == 1


async def test_results_payload_renders_and_gates(tmp_path: Path) -> None:
    llm = FakeLLM([*fixing_steps(), *fixing_steps()])
    cfg = SuiteConfig(
        model=llm,
        rules=RULES,
        system_prompt="s",
        prompt_version="system.v1",
        k=2,
        sandbox_factory=factory(),  # type: ignore[arg-type]
    )
    records = await run_suite(TASKS, cfg)
    payload = results_payload(
        "tasks", llm, "system.v1", summarize(records, 2), [asdict(r) for r in records]
    )
    path = tmp_path / "r.json"
    path.write_text(json.dumps(payload, default=str))

    report = render(path)
    loaded = json.loads(path.read_text())
    worse = json.loads(path.read_text())
    for run in worse["runs"]:
        run["passed"] = False

    assert "| pass@1 | 100.0% [100.0%, 100.0%] |" in report
    assert "revision `rev-1`" in report
    assert gate(loaded, loaded, max_drop=0.1).passed
    assert not gate(loaded, worse, max_drop=0.1).passed

    runner = CliRunner()
    worse_path = tmp_path / "w.json"
    worse_path.write_text(json.dumps(worse))
    same = runner.invoke(cli.app, ["eval", "compare", str(path), str(path)])
    regressed = runner.invoke(cli.app, ["eval", "compare", str(path), str(worse_path)])
    assert same.exit_code == 0
    assert "PASS" in same.output
    assert regressed.exit_code == 1
    assert "FAIL" in regressed.output


def test_unknown_task_id_is_an_error() -> None:
    from agent.evals.tasks import TaskError

    with pytest.raises(TaskError, match="unknown task ids"):
        load_tasks(ROOT / "evals" / "tasks", ids=["nope"])
