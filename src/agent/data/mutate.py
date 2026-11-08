"""Generate bug-fix tasks from a healthy Python repository by AST mutation.

Each mutation flips one small thing (a comparison, an arithmetic operator, a
boolean, an off-by-one in a range or slice, a constant). A mutant becomes a
task only if the repository's tests pass before it and fail after it, so the
oracle is checked when the task is created, not trusted.
"""

from __future__ import annotations

import ast
import random
import shutil
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agent.sandbox.base import ExecResult

_SWAPS: dict[type[ast.AST], type[ast.AST]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.And: ast.Or,
    ast.Or: ast.And,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


@dataclass(frozen=True, slots=True)
class Mutation:
    path: str
    line: int
    kind: str
    before: str
    after: str


def _sites(tree: ast.AST) -> Iterator[tuple[ast.AST, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and type(node.ops[0]) in _SWAPS:
            yield node, "comparison"
        elif isinstance(node, ast.BinOp) and type(node.op) in _SWAPS:
            yield node, "arithmetic"
        elif isinstance(node, ast.BoolOp) and type(node.op) in _SWAPS:
            yield node, "boolean"
        elif isinstance(node, ast.Constant) and isinstance(node.value, bool):
            yield node, "flip-bool"
        elif isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 100:
            yield node, "off-by-one"
        elif isinstance(node, ast.Return) and node.value is not None:
            yield node, "drop-return"


def _apply(node: ast.AST, kind: str) -> None:
    swapped: Any
    if kind == "comparison" and isinstance(node, ast.Compare):
        swapped = _SWAPS[type(node.ops[0])]()
        node.ops[0] = swapped
    elif kind in {"arithmetic", "boolean"} and isinstance(node, ast.BinOp | ast.BoolOp):
        swapped = _SWAPS[type(node.op)]()
        node.op = swapped
    elif kind == "flip-bool" and isinstance(node, ast.Constant):
        node.value = not node.value
    elif kind == "off-by-one" and isinstance(node, ast.Constant) and isinstance(node.value, int):
        node.value = node.value + 1
    elif kind == "drop-return" and isinstance(node, ast.Return):
        node.value = None


def mutate_source(source: str, path: str, rng: random.Random) -> tuple[str, Mutation] | None:
    """One random mutation of one line; None if the file offers no site."""
    tree = ast.parse(source)
    sites = list(_sites(tree))
    if not sites:
        return None
    node, kind = rng.choice(sites)
    line = getattr(node, "lineno", 0)
    lines = source.splitlines(keepends=True)
    before_line = lines[line - 1] if 0 < line <= len(lines) else ""
    segment = ast.get_source_segment(source, node)
    if segment is None or "\n" in segment:
        return None
    _apply(node, kind)
    replacement = ast.unparse(node)
    if kind == "drop-return":
        replacement = "return"
    if replacement == segment or segment not in before_line:
        return None
    after_line = before_line.replace(segment, replacement, 1)
    lines[line - 1] = after_line
    mutated = "".join(lines)
    try:
        ast.parse(mutated)
    except SyntaxError:
        return None
    return mutated, Mutation(path, line, kind, before_line.strip(), after_line.strip())


TestRunner = Callable[[Path], Awaitable[ExecResult]]


async def generate_tasks(
    repo: Path,
    out: Path,
    run_tests: TestRunner,
    *,
    n: int,
    source_glob: str = "src/**/*.py",
    test_command: str = "python -m pytest -q -x",
    seed: int = 0,
    max_attempts_factor: int = 6,
    prefix: str = "mut",
) -> list[Mutation]:
    """Write up to ``n`` verified tasks to ``out/<prefix>-<i>/``."""
    rng = random.Random(seed)  # noqa: S311
    files = [p for p in sorted(repo.glob(source_glob)) if "test" not in p.name]  # noqa: ASYNC240
    if not files:
        raise ValueError(f"no source files match {source_glob} in {repo}")
    baseline = await run_tests(repo)
    if baseline.exit_code != 0:
        raise ValueError(f"tests do not pass on the clean repo:\n{baseline.output[-2000:]}")
    made: list[Mutation] = []
    for _ in range(n * max_attempts_factor):
        if len(made) >= n:
            break
        target = rng.choice(files)
        rel = target.relative_to(repo).as_posix()
        mutated = mutate_source(target.read_text(encoding="utf-8"), rel, rng)
        if mutated is None:
            continue
        text, mutation = mutated
        task_dir = out / f"{prefix}-{len(made):04d}"
        work = task_dir / "repo"
        if task_dir.exists():
            shutil.rmtree(task_dir)
        shutil.copytree(repo, work, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__"))
        (work / rel).write_text(text, encoding="utf-8")
        result = await run_tests(work)
        if result.exit_code == 0:
            shutil.rmtree(task_dir)
            continue
        solution = task_dir / "solution" / rel
        solution.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, solution)
        spec = {
            "id": task_dir.name,
            "instruction": (
                "The test suite fails. Find the bug in the code and fix it. Do not change the "
                f"tests. Run the tests with: {test_command}"
            ),
            "test_command": test_command,
            "tags": ["python", "bugfix", "mutation", mutation.kind, f"source:{repo.name}"],
            "difficulty": "medium",
            "max_turns": 30,
        }
        (task_dir / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        made.append(mutation)
    return made
