"""Single-step tool-call eval: one prompt, one model call, check the first tool call.

Cheap and fast; it catches regressions in tool-call format (schemas, prompt,
chat template, server parser) without running a whole agent task.

    - id: read-named-file
      prompt: Show me what is in config/settings.py.
      expect:                       # one dict, or a list of acceptable ones
        tool: read_file
        args:
          path: {endswith: settings.py}

Argument matchers: a plain value (equality), or one of
``{contains: s}``, ``{endswith: s}``, ``{regex: r}``, ``{any_of: [...]}``.
Arguments not listed are not checked.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.core import ChatModel
from agent.evals.stats import bootstrap_ci, mean_pass_at_k
from agent.llm import LLMError
from agent.tools import ToolRegistry


class CaseError(Exception):
    pass


class Expectation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    prompt: str
    expect: list[Expectation]
    tags: list[str] = Field(default_factory=list)


def load_cases(path: Path) -> list[Case]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CaseError(f"cannot read {path}: {exc}") from exc
    if not isinstance(raw, list):
        raise CaseError(f"{path}: expected a list of cases")
    cases = []
    for entry in raw:
        if isinstance(entry, dict) and isinstance(entry.get("expect"), dict):
            entry = {**entry, "expect": [entry["expect"]]}
        try:
            cases.append(Case.model_validate(entry))
        except ValidationError as exc:
            raise CaseError(f"{path}: invalid case {entry!r}: {exc}") from exc
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise CaseError(f"{path}: duplicate case ids {sorted(duplicates)}")
    return cases


def match_value(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict) and len(expected) == 1:
        ((op, operand),) = expected.items()
        text = "" if actual is None else str(actual)
        if op == "contains":
            return str(operand) in text
        if op == "endswith":
            return text.endswith(str(operand))
        if op == "regex":
            return re.search(str(operand), text) is not None
        if op == "any_of":
            return any(match_value(option, actual) for option in operand)
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip() == actual.strip()
    return bool(expected == actual)


@dataclass(slots=True)
class CaseResult:
    case_id: str
    attempt: int
    outcome: str  # correct | wrong_tool | wrong_args | invalid_json | no_call | model_error
    tool: str | None
    arguments: str | None
    detail: str = ""

    @property
    def correct(self) -> bool:
        return self.outcome == "correct"


def grade(case: Case, tool: str | None, arguments: str | None) -> tuple[str, str]:
    if tool is None:
        return "no_call", ""
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return "invalid_json", exc.msg
    if not isinstance(args, dict):
        return "invalid_json", "arguments are not an object"
    candidates = [e for e in case.expect if e.tool == tool]
    if not candidates:
        return "wrong_tool", f"expected {' or '.join(e.tool for e in case.expect)}"
    for expectation in candidates:
        mismatched = [
            key for key, want in expectation.args.items() if not match_value(want, args.get(key))
        ]
        if not mismatched:
            return "correct", ""
    return "wrong_args", f"mismatched: {', '.join(mismatched)}"


async def run_case(
    case: Case, attempt: int, model: ChatModel, tools: ToolRegistry, system_prompt: str
) -> CaseResult:
    try:
        completion = await model.complete(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": case.prompt},
            ],
            tools=tools.schemas(),
        )
    except LLMError as exc:
        return CaseResult(case.id, attempt, "model_error", None, None, str(exc))
    first = completion.tool_calls[0] if completion.tool_calls else None
    outcome, detail = grade(case, first.name if first else None, first.arguments if first else None)
    return CaseResult(
        case.id,
        attempt,
        outcome,
        first.name if first else None,
        first.arguments if first else None,
        detail or (completion.content[:200] if first is None else ""),
    )


async def run_cases(
    cases: Sequence[Case],
    model: ChatModel,
    tools: ToolRegistry,
    system_prompt: str,
    *,
    k: int = 1,
    concurrency: int = 1,
    progress: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def one(case: Case, attempt: int) -> CaseResult:
        async with semaphore:
            result = await run_case(case, attempt, model, tools, system_prompt)
        if progress is not None:
            progress(result)
        return result

    results = await asyncio.gather(*(one(c, a) for c in cases for a in range(1, k + 1)))
    return sorted(results, key=lambda r: (r.case_id, r.attempt))


def summarize_cases(results: Sequence[CaseResult]) -> dict[str, Any]:
    outcomes: dict[str, list[bool]] = {}
    for result in results:
        outcomes.setdefault(result.case_id, []).append(result.correct)
    accuracy = bootstrap_ci(outcomes, lambda o: mean_pass_at_k(o, 1))
    counts = Counter(r.outcome for r in results)
    calls = [r for r in results if r.tool is not None]
    return {
        "cases": len(outcomes),
        "runs": len(results),
        "accuracy": asdict(accuracy),
        "outcomes": dict(sorted(counts.items())),
        "invalid_json_rate": counts["invalid_json"] / len(calls) if calls else None,
        "no_call_rate": counts["no_call"] / len(results),
        "failures": [asdict(r) for r in results if not r.correct],
    }
