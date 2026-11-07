"""OpenTelemetry setup. Without ``configure`` every span is a no-op.

Span tree:  invoke_agent → agent.turn → chat {model} | execute_tool {name}

Spans carry metrics, identifiers and decisions, never message content: the
conversation contains user code and command output, which may hold secrets.
Full trajectories already live in the results JSON of headless runs and evals.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from agent.telemetry.redact import redact_value

SCOPE = "agent"
DEFAULT_TRACE_FILE = Path.home() / ".local" / "state" / "agent" / "traces.jsonl"

AttributeValue = str | bool | int | float | Sequence[str] | Sequence[int] | Sequence[float]


class Exporter(StrEnum):
    NONE = "none"
    FILE = "file"
    CONSOLE = "console"
    OTLP = "otlp"  # endpoint from OTEL_EXPORTER_OTLP_ENDPOINT, e.g. Phoenix


def tracer() -> trace.Tracer:
    return trace.get_tracer(SCOPE)


def set_attributes(span: trace.Span, attributes: Mapping[str, Any]) -> None:
    """The only way attributes get onto spans here, so redaction cannot be skipped."""
    for key, value in attributes.items():
        if value is None:
            continue
        span.set_attribute(key, redact_value(value))


def current_trace_id() -> str | None:
    context = trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def span_record(span: ReadableSpan) -> dict[str, Any]:
    context = span.get_span_context()
    return {
        "name": span.name,
        "trace_id": format(context.trace_id, "032x") if context else None,
        "span_id": format(context.span_id, "016x") if context else None,
        "parent_id": format(span.parent.span_id, "016x") if span.parent else None,
        "start_ns": span.start_time,
        "end_ns": span.end_time,
        "status": span.status.status_code.name,
        "status_message": span.status.description,
        "attributes": dict(span.attributes or {}),
    }


class JsonlSpanExporter(SpanExporter):
    """One span per line; easy to grep, to load, and to replay into a real backend."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        lines = [json.dumps(span_record(s), default=str) for s in spans]
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write("".join(line + "\n" for line in lines))
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


def configure(
    exporter: Exporter, *, path: Path = DEFAULT_TRACE_FILE, service_version: str = "0.1.0"
) -> TracerProvider | None:
    """Install a global tracer provider. Call once per process; returns it for shutdown()."""
    if exporter is Exporter.NONE:
        return None
    provider = TracerProvider(
        resource=Resource.create({"service.name": "agent", "service.version": service_version})
    )
    if exporter is Exporter.FILE:
        provider.add_span_processor(BatchSpanProcessor(JsonlSpanExporter(path)))
    elif exporter is Exporter.CONSOLE:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider
