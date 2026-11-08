"""Turn eval-style runs of a teacher model into an SFT dataset.

Pipeline: run the harness headless with the teacher (the same runner as
evals, so trajectories are in this harness's tool format) → keep runs whose
tests pass (rejection sampling) → filters → decontamination against every
eval task → MinHash dedup → JSONL in OpenAI chat format with tools.
Failed runs of tasks that also have a passing run become DPO pairs.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.evals.tasks import Task
from agent.telemetry.redact import redact

NGRAM = 13
MINHASH_PERMS = 64
DEDUP_THRESHOLD = 0.85


@dataclass(frozen=True, slots=True)
class Trajectory:
    task_id: str
    attempt: int
    passed: bool
    stop_reason: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    stats: dict[str, Any]


@dataclass
class FilterReport:
    seen: int = 0
    kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)

    def drop(self, reason: str) -> None:
        self.dropped[reason] += 1


def _text(messages: Sequence[dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            parts.append(content)
        for call in m.get("tool_calls") or []:
            parts.append(call["function"]["arguments"])
    return "\n".join(parts)


def ngrams(text: str, n: int = NGRAM) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i : i + n]) for i in range(max(0, len(words) - n + 1))}


def eval_fingerprints(tasks: Iterable[Task]) -> tuple[set[str], set[str]]:
    """Task ids and 13-grams of every eval task's instruction and source files."""
    ids: set[str] = set()
    grams: set[str] = set()
    for task in tasks:
        ids.add(task.id)
        grams |= ngrams(task.instruction)
        for path in task.repo.rglob("*.py"):
            grams |= ngrams(path.read_text(encoding="utf-8", errors="replace"))
    return ids, grams


def _minhash(text: str, seeds: Sequence[int]) -> tuple[int, ...]:
    shingles = ngrams(text, 5) or {text}
    return tuple(
        min(
            int.from_bytes(hashlib.blake2b(f"{s}:{g}".encode(), digest_size=8).digest(), "big")
            for g in shingles
        )
        for s in seeds
    )


def _similarity(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def select(
    trajectories: Sequence[Trajectory],
    *,
    eval_ids: set[str],
    eval_grams: set[str],
    max_turns: int = 40,
    max_chars: int = 200_000,
    contamination_overlap: float = 0.05,
) -> tuple[list[Trajectory], FilterReport]:
    report = FilterReport()
    seeds = list(range(MINHASH_PERMS))
    kept: list[Trajectory] = []
    signatures: list[tuple[int, ...]] = []
    for traj in trajectories:
        report.seen += 1
        text = _text(traj.messages)
        if not traj.passed:
            report.drop("tests_failed")
        elif traj.stop_reason != "completed":
            report.drop(f"stop_{traj.stop_reason}")
        elif traj.stats.get("invalid_tool_calls", 0):
            report.drop("invalid_tool_calls")
        elif traj.stats.get("turns", 0) > max_turns or len(text) > max_chars:
            report.drop("too_long")
        elif redact(text) != text:
            report.drop("secret_or_pii")
        elif traj.task_id in eval_ids:
            report.drop("eval_task")
        elif (grams := ngrams(text)) and len(grams & eval_grams) / len(
            grams
        ) > contamination_overlap:
            report.drop("eval_overlap")
        else:
            signature = _minhash(text, seeds)
            if any(_similarity(signature, s) >= DEDUP_THRESHOLD for s in signatures):
                report.drop("near_duplicate")
                continue
            signatures.append(signature)
            kept.append(traj)
            report.kept += 1
    return kept, report


def split_by_repo(
    trajectories: Sequence[Trajectory],
    source_of: dict[str, str],
    val_fraction: float,
    seed: int = 0,
) -> tuple[list[Trajectory], list[Trajectory]]:
    """Train/val split by source repository, never by task: tasks mutated from
    the same repository share code, so a task-level split leaks."""
    repos = sorted({source_of.get(t.task_id, t.task_id) for t in trajectories})
    rng = random.Random(seed)  # noqa: S311
    rng.shuffle(repos)
    n_val = max(1, round(len(repos) * val_fraction)) if len(repos) > 1 else 0
    val_repos = set(repos[:n_val])
    train = [t for t in trajectories if source_of.get(t.task_id, t.task_id) not in val_repos]
    val = [t for t in trajectories if source_of.get(t.task_id, t.task_id) in val_repos]
    return train, val


def dpo_pairs(trajectories: Sequence[Trajectory]) -> list[dict[str, Any]]:
    """(prompt, chosen, rejected) per task that has both a pass and a fail."""
    by_task: dict[str, list[Trajectory]] = {}
    for t in trajectories:
        by_task.setdefault(t.task_id, []).append(t)
    pairs = []
    for task_id, runs in sorted(by_task.items()):
        good = [r for r in runs if r.passed and r.stop_reason == "completed"]
        bad = [r for r in runs if not r.passed]
        if good and bad:
            prompt = good[0].messages[:2]
            pairs.append(
                {
                    "task_id": task_id,
                    "prompt": prompt,
                    "chosen": good[0].messages[2:],
                    "rejected": bad[0].messages[2:],
                    "tools": good[0].tools,
                }
            )
    return pairs


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def sft_row(t: Trajectory) -> dict[str, Any]:
    return {"messages": t.messages, "tools": t.tools, "task_id": t.task_id}
