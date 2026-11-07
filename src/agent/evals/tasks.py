"""Golden-set task format: ``evals/tasks/<id>/task.yaml`` plus a repo snapshot.

    id: off-by-one-moving-average
    instruction: The test suite fails. Fix the bug; do not change the tests.
    test_command: python -m pytest -q
    protected: ["test_*.py"]   # restored from the snapshot before grading
    tags: [python, bugfix]
    difficulty: easy
    max_turns: 20

Layout:  repo/    what the agent sees
         hidden/  optional files copied over the workspace before grading
                  (extra tests the agent never saw, to catch overfitting)
"""

from __future__ import annotations

import fnmatch
import shutil
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class TaskError(Exception):
    pass


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    instruction: str
    test_command: str
    protected: list[str] = Field(
        default_factory=lambda: ["test_*.py", "*/test_*.py", "tests/*", "conftest.py", "pytest.ini"]
    )
    tags: list[str] = Field(default_factory=list)
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    max_turns: int = Field(default=20, ge=1)
    test_timeout_s: int = Field(default=120, ge=1)
    root: Path = Field(exclude=True)

    @property
    def repo(self) -> Path:
        return self.root / "repo"

    @property
    def hidden(self) -> Path:
        return self.root / "hidden"


def load_task(path: Path) -> Task:
    spec = path / "task.yaml"
    try:
        raw = yaml.safe_load(spec.read_text(encoding="utf-8"))
        task = Task(root=path, **raw)
    except FileNotFoundError as exc:
        raise TaskError(f"{spec} not found") from exc
    except (yaml.YAMLError, ValidationError, TypeError) as exc:
        raise TaskError(f"{spec} is invalid: {exc}") from exc
    if task.id != path.name:
        raise TaskError(f"{spec}: id {task.id!r} must match directory name {path.name!r}")
    if not task.repo.is_dir():
        raise TaskError(f"{task.id}: missing repo/ directory")
    return task


def load_tasks(
    root: Path, *, tags: list[str] | None = None, ids: list[str] | None = None
) -> list[Task]:
    tasks = [load_task(p) for p in sorted(root.iterdir()) if (p / "task.yaml").is_file()]
    if ids:
        missing = set(ids) - {t.id for t in tasks}
        if missing:
            raise TaskError(f"unknown task ids: {', '.join(sorted(missing))}")
        tasks = [t for t in tasks if t.id in ids]
    if tags:
        tasks = [t for t in tasks if set(tags) & set(t.tags)]
    return tasks


def _is_protected(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(rel, p) for p in patterns)


def prepare_workspace(task: Task, dest: Path) -> Path:
    """Fresh copy of the task repo for one run."""
    shutil.copytree(task.repo, dest, symlinks=False)
    return dest


def prepare_grading(task: Task, workspace: Path) -> list[str]:
    """Restore protected files from the snapshot and overlay hidden files.

    Returns the protected files the agent had changed or deleted, which the
    report shows: a run that edits the tests is suspicious even if it passes.
    """
    tampered = []
    for src in sorted(task.repo.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(task.repo).as_posix()
        if not _is_protected(rel, task.protected):
            continue
        dst = workspace / rel
        if not dst.is_file() or dst.read_bytes() != src.read_bytes():
            tampered.append(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    if task.hidden.is_dir():
        shutil.copytree(task.hidden, workspace, dirs_exist_ok=True)
    return tampered
