"""Retrieval eval, measured separately from generation.

Golden set from git history: a commit's subject line is the query, the Python
files it changed (and that still exist) are the relevant set. File-level
metrics, because "which file do I open" is what the agent needs from search.

Caveat: the index is built at HEAD, after those commits landed, so a commit
that introduced a new identifier makes its query easier than it was at the
time. That biases all methods alike and does not change their ranking much.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agent.evals.stats import bootstrap_mean
from agent.rag.search import CANDIDATES, Method, Searcher

_SKIP = re.compile(r"^(merge|bump|release|version|revert|wip\b|update changelog)", re.I)


@dataclass(frozen=True, slots=True)
class RetrievalCase:
    commit: str
    query: str
    relevant: frozenset[str]


def _git(repo: Path, *args: str) -> str:
    cmd = ["git", *args]
    return subprocess.run(cmd, cwd=repo, capture_output=True, text=True, check=True).stdout  # noqa: S603


def cases_from_git(repo: Path, max_commits: int = 300, max_files: int = 5) -> list[RetrievalCase]:
    log = _git(
        repo, "log", "--no-merges", f"-n{max_commits * 3}", "--name-only", "--format=%x00%H%x1f%s"
    )
    existing = set(_git(repo, "ls-files").split())
    cases = []
    for block in log.split("\0")[1:]:
        header, _, files = block.partition("\n")
        sha, _, subject = header.partition("\x1f")
        changed = {f for f in files.split() if f.endswith(".py") and f in existing}
        if not 1 <= len(changed) <= max_files or len(subject) < 12 or _SKIP.match(subject):
            continue
        cases.append(RetrievalCase(sha[:10], subject.strip(), frozenset(changed)))
        if len(cases) >= max_commits:
            break
    return cases


def ranked_files(paths: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for p in paths:
        if p not in seen:
            seen.append(p)
    return seen


def case_metrics(files: Sequence[str], relevant: frozenset[str]) -> dict[str, float]:
    def recall(k: int) -> float:
        return len(set(files[:k]) & relevant) / len(relevant)

    rr = next((1.0 / (i + 1) for i, f in enumerate(files) if f in relevant), 0.0)
    return {"recall@5": recall(5), "recall@10": recall(10), "mrr": rr}


async def evaluate(
    searcher: Searcher, cases: Sequence[RetrievalCase], methods: Sequence[Method] | None = None
) -> dict[str, Any]:
    methods = list(methods or searcher.methods)
    summary: dict[str, Any] = {"cases": len(cases), "methods": {}}
    runs: list[dict[str, Any]] = []
    for method in methods:
        per_case: dict[str, dict[str, float]] = {}
        for case in cases:
            hits = await searcher.search(case.query, CANDIDATES, method)
            files = ranked_files([h.chunk.path for h in hits])
            metrics = case_metrics(files, case.relevant)
            per_case[case.commit] = metrics
            runs.append(
                {
                    "method": method.value,
                    **asdict(case),
                    "relevant": sorted(case.relevant),
                    "top_files": files[:10],
                    **metrics,
                }
            )
        means = {
            name: sum(m[name] for m in per_case.values()) / len(per_case) if per_case else 0.0
            for name in ("recall@5", "recall@10", "mrr")
        }
        ci = bootstrap_mean([m["recall@10"] for m in per_case.values()]) if per_case else None
        summary["methods"][method.value] = {**means, "recall@10_ci": asdict(ci) if ci else None}
    return {"summary": summary, "runs": runs}


def render_retrieval(data: dict[str, Any], source: str) -> str:
    s = data["summary"]
    lines = [
        f"# retrieval eval · {data['repo']} · embedder `{data.get('embedder') or 'none'}`",
        "",
        f"- source: `{source}` · created {data['created']} · git `{data['git']}`",
        f"- {s['cases']} queries from git history (commit subject → changed .py files);"
        f" index: {data['chunks']} chunks from {data['files']} files",
        "",
        "| method | recall@5 | recall@10 [95% CI] | MRR |",
        "|---|---|---|---|",
    ]
    for method, m in s["methods"].items():
        ci = m["recall@10_ci"]
        ci_text = f" [{ci['low']:.1%}, {ci['high']:.1%}]" if ci else ""
        lines.append(
            f"| {method} | {m['recall@5']:.1%} | {m['recall@10']:.1%}{ci_text} | {m['mrr']:.3f} |"
        )
    return "\n".join(lines) + "\n"
