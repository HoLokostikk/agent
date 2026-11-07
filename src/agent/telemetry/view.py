"""Read the JSONL trace file: list runs, show one trace as a tree, aggregate metrics."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int
    status: str
    attributes: dict[str, Any]

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9


def read_spans(path: Path) -> Iterator[Span]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield Span(
                name=raw["name"],
                trace_id=raw["trace_id"],
                span_id=raw["span_id"],
                parent_id=raw.get("parent_id"),
                start_ns=int(raw["start_ns"]),
                end_ns=int(raw["end_ns"]),
                status=raw.get("status", "UNSET"),
                attributes=raw.get("attributes", {}),
            )


def runs(path: Path, limit: int = 20) -> list[Span]:
    roots = [s for s in read_spans(path) if s.name == "invoke_agent"]
    return sorted(roots, key=lambda s: s.start_ns, reverse=True)[:limit]


_SHOWN = (
    "agent.stop_reason",
    "agent.turn",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "agent.cost_usd",
    "agent.ttft_s",
    "agent.attempts",
    "error.type",
    "agent.permission.verdict",
    "process.exit.code",
    "agent.tool.command",
)


def render_tree(path: Path, trace_id: str) -> str:
    spans = [s for s in read_spans(path) if s.trace_id.startswith(trace_id)]
    if not spans:
        return f"no spans for trace {trace_id} in {path}"
    children: dict[str | None, list[Span]] = defaultdict(list)
    ids = {s.span_id for s in spans}
    for s in spans:
        parent = s.parent_id if s.parent_id in ids else None
        children[parent].append(s)
    lines = [f"trace {spans[0].trace_id}"]

    def walk(parent: str | None, depth: int) -> None:
        for s in sorted(children[parent], key=lambda x: x.start_ns):
            mark = "✗" if s.status == "ERROR" else "·"
            details = " ".join(
                f"{k.split('.')[-1]}={s.attributes[k]}" for k in _SHOWN if k in s.attributes
            )
            lines.append(f"{'  ' * depth}{mark} {s.name} {s.duration_s:.2f}s {details}".rstrip())
            walk(s.span_id, depth + 1)

    walk(None, 0)
    return "\n".join(lines)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def stats(path: Path) -> dict[str, Any]:
    """p50/p95 latency and TTFT per model, cost per (successful) run, tool failure rates."""
    chat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    tool_calls: Counter[str] = Counter()
    tool_failures: Counter[str] = Counter()
    run_costs: list[float] = []
    completed_costs: list[float] = []
    stop_reasons: Counter[str] = Counter()
    for s in read_spans(path):
        a = s.attributes
        if a.get("gen_ai.operation.name") == "chat":
            m = chat[str(a.get("agent.model.alias", "?"))]
            m["latency_s"].append(s.duration_s)
            if "agent.ttft_s" in a:
                m["ttft_s"].append(float(a["agent.ttft_s"]))
            if "agent.output_tokens_per_s" in a:
                m["tokens_per_s"].append(float(a["agent.output_tokens_per_s"]))
        elif a.get("gen_ai.operation.name") == "execute_tool":
            name = str(a.get("gen_ai.tool.name"))
            tool_calls[name] += 1
            if not a.get("agent.tool.ok", True):
                tool_failures[name] += 1
        elif s.name == "invoke_agent":
            reason = str(a.get("agent.stop_reason", "unknown"))
            stop_reasons[reason] += 1
            if "agent.cost_usd" in a:
                run_costs.append(float(a["agent.cost_usd"]))
                if reason == "completed":
                    completed_costs.append(float(a["agent.cost_usd"]))
    models = {
        alias: {
            "calls": len(m["latency_s"]),
            **{
                f"{metric}_{label}": _percentile(values, q)
                for metric, values in m.items()
                for label, q in (("p50", 0.5), ("p95", 0.95))
            },
        }
        for alias, m in chat.items()
    }
    completed = stop_reasons.get("completed", 0)
    return {
        "models": models,
        "runs": sum(stop_reasons.values()),
        "stop_reasons": dict(stop_reasons),
        "cost_per_run_usd": sum(run_costs) / len(run_costs) if run_costs else None,
        # "completed" is the agent's own claim; tests decide success in evals.
        "cost_per_completed_run_usd": sum(run_costs) / completed if completed else None,
        "tools": {
            name: {"calls": n, "failure_rate": tool_failures[name] / n}
            for name, n in sorted(tool_calls.items())
        },
    }
