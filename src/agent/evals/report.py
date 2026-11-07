"""Markdown reports from results JSON, and the regression gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.evals.stats import Interval, mean_pass_at_k, paired_bootstrap_diff


def load_results(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported schema_version {data.get('schema_version')}")
    return data


def _pct(interval: dict[str, Any]) -> str:
    return f"{interval['estimate']:.1%} [{interval['low']:.1%}, {interval['high']:.1%}]"


def _money(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def _header(data: dict[str, Any], source: str) -> list[str]:
    m = data["model"]
    return [
        f"# {data['kind']} eval · {m['alias']} · {data['prompt_version']}",
        "",
        f"- source: `{source}`",
        f"- created: {data['created']} · git `{data['git']}`",
        f"- model: `{m['model']}` · revision `{m['revision']}`",
        f"- sampling: `{json.dumps(m.get('sampling', {}))}`",
        "",
    ]


def render_tasks(data: dict[str, Any], source: str) -> str:
    s = data["summary"]
    k = s["k"]
    lines = _header(data, source)
    lines += [
        f"{s['tasks']} tasks × {k} runs = {s['runs']} runs. Intervals: 95% percentile "
        "bootstrap over tasks.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| pass@1 | {_pct(s['pass_at_1'])} |",
        f"| pass^{k} | {_pct(s[f'pass_hat_{k}'])} |",
        f"| pass@{k} | {s[f'pass_at_{k}']:.1%} |",
        f"| mean turns | {s['mean_turns']:.1f} |",
        f"| mean tokens in / out | {s['mean_input_tokens']:.0f} / {s['mean_output_tokens']:.0f} |",
        f"| mean wall time | {s['mean_wall_s']:.1f} s |",
        f"| cost per task | {_money(s['cost_per_task_usd'])} |",
        f"| cost per successful task | {_money(s['cost_per_successful_task_usd'])} |",
        "| invalid tool call rate | "
        + ("—" if s["invalid_tool_call_rate"] is None else f"{s['invalid_tool_call_rate']:.1%}")
        + f" of {s['tool_calls']} calls |",
        f"| runs that edited tests | {s['tampered_runs']} |",
        f"| infrastructure errors | {s['infra_errors']} |",
        "",
        "Stop reasons: "
        + ", ".join(f"{reason} {n}" for reason, n in sorted(s["stop_reasons"].items())),
        "",
        "| tool | calls | error rate |",
        "|---|---|---|",
    ]
    for name, row in s["tool_error_rates"].items():
        lines.append(f"| {name} | {row['calls']} | {row['error_rate']:.0%} |")
    lines += ["", "| task | passed |", "|---|---|"]
    for task, row in s["per_task"].items():
        lines.append(f"| {task} | {row['passed']}/{row['runs']} |")
    failed = [r for r in data["runs"] if not r["passed"]]
    if failed:
        lines += [
            "",
            "Failed runs (open with `agent trace show <trace>`):",
            "",
            "| task | # | stop reason | turns | trace |",
            "|---|---|---|---|---|",
        ]
        for r in failed:
            trace = f"`{r['trace_id']}`" if r.get("trace_id") else "—"
            turns = r["stats"].get("turns", "—") if r["stats"] else "—"
            lines.append(
                f"| {r['task_id']} | {r['attempt']} | {r['stop_reason']} | {turns} | {trace} |"
            )
    return "\n".join(lines) + "\n"


def render_toolcalls(data: dict[str, Any], source: str) -> str:
    s = data["summary"]
    lines = _header(data, source)
    lines += [
        f"{s['cases']} cases, {s['runs']} runs.",
        "",
        "| metric | value |",
        "|---|---|",
        f"| accuracy (right tool and arguments) | {_pct(s['accuracy'])} |",
        "| invalid JSON rate | "
        + ("—" if s["invalid_json_rate"] is None else f"{s['invalid_json_rate']:.1%}")
        + " |",
        f"| no tool call | {s['no_call_rate']:.1%} |",
        "",
        "Outcomes: " + ", ".join(f"{k} {v}" for k, v in s["outcomes"].items()),
        "",
        "| case | outcome | tool | arguments |",
        "|---|---|---|---|",
    ]
    for f in s["failures"]:
        args = (f["arguments"] or "").replace("|", "\\|")[:80]
        lines.append(f"| {f['case_id']} | {f['outcome']} | {f['tool'] or '—'} | `{args}` |")
    return "\n".join(lines) + "\n"


def render(path: Path) -> str:
    data = load_results(path)
    if data["kind"] == "tasks":
        return render_tasks(data, path.name)
    if data["kind"] == "toolcalls":
        return render_toolcalls(data, path.name)
    raise ValueError(f"unknown results kind {data['kind']!r}")


def _outcomes(data: dict[str, Any]) -> dict[str, list[bool]]:
    outcomes: dict[str, list[bool]] = {}
    key = "task_id" if data["kind"] == "tasks" else "case_id"
    for run in data["runs"]:
        passed = run["passed"] if data["kind"] == "tasks" else run["outcome"] == "correct"
        outcomes.setdefault(run[key], []).append(bool(passed))
    return outcomes


@dataclass(frozen=True, slots=True)
class GateResult:
    passed: bool
    diff: Interval
    max_drop: float
    message: str


def gate(baseline: dict[str, Any], candidate: dict[str, Any], *, max_drop: float) -> GateResult:
    """Fail if the candidate's pass rate is more than ``max_drop`` below baseline.

    The point estimate decides; the paired 95% CI is reported alongside so a
    reviewer can see whether the drop is distinguishable from noise.
    """
    if baseline["kind"] != candidate["kind"]:
        raise ValueError("cannot compare results of different kinds")
    diff = paired_bootstrap_diff(
        _outcomes(baseline), _outcomes(candidate), lambda o: mean_pass_at_k(o, 1)
    )
    ok = diff.estimate >= -max_drop
    verdict = "PASS" if ok else "FAIL"
    message = (
        f"{verdict}: pass@1 change {diff.estimate:+.1%} "
        f"(95% CI [{diff.low:+.1%}, {diff.high:+.1%}]), allowed drop {max_drop:.1%}"
    )
    return GateResult(ok, diff, max_drop, message)
