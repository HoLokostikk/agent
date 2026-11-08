import random
from pathlib import Path
from typing import Any

import pytest

from agent.data.collect import (
    Trajectory,
    dpo_pairs,
    eval_fingerprints,
    ngrams,
    select,
    split_by_repo,
)
from agent.data.mutate import generate_tasks, mutate_source
from agent.evals.tasks import load_task, load_tasks
from agent.sandbox import ExecResult

SOURCE = """def clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def total(items):
    return sum(items) + 0
"""


def test_mutation_changes_exactly_one_line_and_parses() -> None:
    for seed in range(20):
        result = mutate_source(SOURCE, "m.py", random.Random(seed))
        if result is None:
            continue
        text, mutation = result
        changed = [a for a, b in zip(SOURCE.splitlines(), text.splitlines(), strict=True) if a != b]
        assert len(changed) == 1
        assert mutation.before != mutation.after
        compile(text, "m.py", "exec")


def test_file_without_sites_gives_none() -> None:
    assert mutate_source("import os\n", "m.py", random.Random(0)) is None


async def test_generate_tasks_keeps_only_detected_mutants(tmp_path: Path) -> None:
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "m.py").write_text(SOURCE)
    original = SOURCE
    detected = {"n": 0}

    async def run_tests(path: Path) -> ExecResult:
        text = (path / "src" / "m.py").read_text()
        if text == original:
            return ExecResult(0, "passed")
        changed = [
            i
            for i, (a, b) in enumerate(zip(original.splitlines(), text.splitlines(), strict=True))
            if a != b
        ]
        if changed and changed[0] < 7:
            detected["n"] += 1
            return ExecResult(1, "failed")
        return ExecResult(0, "passed")

    made = await generate_tasks(repo, tmp_path / "tasks", run_tests, n=3, seed=1, prefix="mut-proj")

    tasks = load_tasks(tmp_path / "tasks")
    assert len(made) == len(tasks) == 3
    assert all(m.line < 7 for m in made)
    task = load_task(tmp_path / "tasks" / "mut-proj-0000")
    assert "source:proj" in task.tags
    assert (task.root / "solution" / "src" / "m.py").read_text() == original


async def test_generate_refuses_a_repo_with_failing_tests(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text(SOURCE)

    async def failing(path: Path) -> ExecResult:
        return ExecResult(1, "broken")

    with pytest.raises(ValueError, match="do not pass"):
        await generate_tasks(tmp_path, tmp_path / "out", failing, n=1)


def traj(task: str, passed: bool = True, text: str = "fixed the bug", **stats: Any) -> Trajectory:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": f"task {task}"},
        {"role": "assistant", "content": f"{text} in {task}"},
    ]
    return Trajectory(task, 1, passed, "completed", messages, [], {"turns": 3, **stats})


def test_select_applies_every_filter() -> None:
    eval_text = " ".join(f"w{i}" for i in range(40))
    body = " ".join(f"step{i} read file and edit line" for i in range(30))
    trajectories = [
        traj("a", text=body),
        traj("b", passed=False),
        traj("c", invalid_tool_calls=1),
        traj("d", text="key sk-or-v1-0123456789abcdef0123456789"),
        traj("evaltask"),
        traj("e", text=eval_text),
        traj("a2", text=body),
    ]

    kept, report = select(trajectories, eval_ids={"evaltask"}, eval_grams=ngrams(eval_text))

    assert [t.task_id for t in kept] == ["a"]
    assert report.dropped == {
        "tests_failed": 1,
        "invalid_tool_calls": 1,
        "secret_or_pii": 1,
        "eval_task": 1,
        "eval_overlap": 1,
        "near_duplicate": 1,
    }


def test_split_is_by_source_repo() -> None:
    ts = [traj(f"r{r}-{i}", text=f"unique {r} {i} " * 10) for r in range(5) for i in range(3)]
    source = {t.task_id: t.task_id.split("-")[0] for t in ts}

    train, val = split_by_repo(ts, source, 0.2)

    assert {source[t.task_id] for t in train}.isdisjoint({source[t.task_id] for t in val})
    assert len(val) == 3


def test_dpo_pairs_need_a_pass_and_a_fail() -> None:
    pairs = dpo_pairs([traj("a"), traj("a", passed=False), traj("b")])

    assert [p["task_id"] for p in pairs] == ["a"]
    assert pairs[0]["prompt"][1]["content"] == "task a"


def test_eval_fingerprints_cover_ids_and_code() -> None:
    root = Path(__file__).parents[2] / "evals" / "tasks"
    ids, grams = eval_fingerprints(load_tasks(root))

    assert "wrong-operator-add" in ids
    assert grams
