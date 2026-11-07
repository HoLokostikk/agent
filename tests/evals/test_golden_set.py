"""Every golden task must be a valid oracle: fails as shipped, passes when fixed.

A task whose tests pass without a fix, or fail even with the reference
solution, silently corrupts every pass rate measured on the set.
"""

import shutil
from pathlib import Path

import pytest

from agent.evals.tasks import Task, load_tasks, prepare_grading, prepare_workspace
from agent.sandbox import DockerSandbox, build_image, image_exists
from agent.tools.files import IGNORED_DIRS

TASKS = load_tasks(Path(__file__).parents[2] / "evals" / "tasks")
IDS = [t.id for t in TASKS]


def test_golden_set_is_not_empty() -> None:
    assert len(TASKS) >= 10


@pytest.mark.parametrize("task", TASKS, ids=IDS)
def test_task_is_well_formed(task: Task) -> None:
    solution = task.root / "solution"
    assert solution.is_dir(), "every task ships a reference solution"
    for path in solution.rglob("*"):
        if path.is_file():
            rel = path.relative_to(solution)
            assert (task.repo / rel).is_file(), f"solution patches a file not in repo: {rel}"
    for test_file in task.repo.rglob("test_*.py"):
        test_rel = test_file.relative_to(task.repo).as_posix()
        assert test_rel in _protected(task), f"{test_rel} is not protected; the agent could edit it"


def _protected(task: Task) -> set[str]:
    import fnmatch

    return {
        p.relative_to(task.repo).as_posix()
        for p in task.repo.rglob("*")
        if p.is_file()
        and any(
            fnmatch.fnmatchcase(p.relative_to(task.repo).as_posix(), pat) for pat in task.protected
        )
    }


def test_task_ids_are_unique_and_tagged() -> None:
    assert len(set(IDS)) == len(IDS)
    for task in TASKS:
        assert task.tags, task.id
        assert not any(part in IGNORED_DIRS for part in task.repo.parts)


def test_grading_restores_edited_tests(tmp_path: Path) -> None:
    task = next(t for t in TASKS if t.id == "wrong-operator-add")
    ws = prepare_workspace(task, tmp_path / "ws")
    (ws / "test_calc.py").write_text("def test_nothing() -> None:\n    pass\n")

    tampered = prepare_grading(task, ws)

    assert tampered == ["test_calc.py"]
    assert (ws / "test_calc.py").read_text() == (task.repo / "test_calc.py").read_text()


def test_grading_overlays_hidden_tests(tmp_path: Path) -> None:
    task = next(t for t in TASKS if t.hidden.is_dir())
    ws = prepare_workspace(task, tmp_path / "ws")

    prepare_grading(task, ws)

    for hidden in task.hidden.rglob("*.py"):
        assert (ws / hidden.relative_to(task.hidden)).is_file()


async def _grade(task: Task, ws: Path) -> int:
    prepare_grading(task, ws)
    async with DockerSandbox(ws) as box:
        result = await box.exec(task.test_command, task.test_timeout_s)
    return result.exit_code


@pytest.mark.docker
@pytest.mark.parametrize("task", TASKS, ids=IDS)
async def test_task_fails_as_shipped_and_passes_with_solution(task: Task, tmp_path: Path) -> None:
    if not await image_exists():
        await build_image()

    broken = prepare_workspace(task, tmp_path / "broken")
    assert await _grade(task, broken) != 0, "tests pass without any fix"

    fixed = prepare_workspace(task, tmp_path / "fixed")
    shutil.copytree(task.root / "solution", fixed, dirs_exist_ok=True)
    assert await _grade(task, fixed) == 0, "reference solution does not pass"
