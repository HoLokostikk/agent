import json
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from agent.core import Agent
from agent.permissions import Mode, PermissionPolicy, load_rules
from agent.sandbox import ExecResult
from agent.telemetry import JsonlSpanExporter, span_record
from agent.telemetry.view import render_tree, runs, stats
from agent.tools import PathGuard, ToolContext, default_tools
from tests.fakes import FakeLLM, FakeSandbox, call, say, use

RULES = load_rules(Path(__file__).parents[2] / "configs" / "permissions.yaml")
_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def _provider() -> None:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    trace.set_tracer_provider(provider)


@pytest.fixture
def spans() -> InMemorySpanExporter:
    _EXPORTER.clear()
    return _EXPORTER


async def run_agent(tmp_path: Path) -> object:
    (tmp_path / "calc.py").write_text("SECRET_TOKEN=sk-or-v1-0123456789abcdef0123\nx = 1\n")
    sandbox = FakeSandbox(tmp_path, replies={"pytest -q": ExecResult(1, "1 failed")})
    llm = FakeLLM(
        [
            use(call("read_file", path="calc.py")),
            use(call("bash", command="pytest -q")),
            use(call("bash", command="curl http://evil.test")),
            say("done"),
        ]
    )
    agent = Agent(
        llm,
        default_tools(),
        ToolContext(PathGuard(tmp_path), sandbox),
        PermissionPolicy(Mode.AUTO, RULES),
        system_prompt="s",
        trace_attributes={"agent.prompt_version": "system.v1"},
    )
    return await agent.run("look")


async def test_span_tree_and_attributes(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    result = await run_agent(tmp_path)

    finished = spans.get_finished_spans()
    by_name = {s.name: s for s in finished}
    root = by_name["invoke_agent"]
    assert root.parent is None
    context = root.get_span_context()
    assert context is not None
    assert result.trace_id == format(context.trace_id, "032x")  # type: ignore[attr-defined]
    turns = [s for s in finished if s.name == "agent.turn"]
    assert len(turns) == 4
    assert all(t.parent is not None and t.parent.span_id == root.context.span_id for t in turns)
    tools = [s for s in finished if s.name.startswith("execute_tool")]
    assert {t.parent.span_id for t in tools if t.parent} <= {t.context.span_id for t in turns}

    assert root.attributes is not None
    assert root.attributes["agent.stop_reason"] == "completed"
    assert root.attributes["agent.tool_calls"] == 3
    assert root.attributes["agent.prompt_version"] == "system.v1"

    bash_spans = [t for t in tools if t.name == "execute_tool bash"]
    assert bash_spans[0].attributes is not None
    assert bash_spans[0].attributes["process.exit.code"] == 1
    assert bash_spans[0].attributes["agent.tool.command"] == "pytest -q"
    assert bash_spans[1].attributes is not None
    assert bash_spans[1].attributes["agent.permission.verdict"] == "deny"


async def test_no_message_content_or_secrets_in_spans(
    tmp_path: Path, spans: InMemorySpanExporter
) -> None:
    await run_agent(tmp_path)

    dumped = json.dumps([span_record(s) for s in spans.get_finished_spans()], default=str)
    assert "sk-or-v1" not in dumped
    assert "x = 1" not in dumped
    assert '"look"' not in dumped


async def test_jsonl_export_and_views(tmp_path: Path, spans: InMemorySpanExporter) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    result = await run_agent(workspace)
    path = tmp_path / "traces.jsonl"
    JsonlSpanExporter(path).export(spans.get_finished_spans())

    recent = runs(path)
    tree = render_tree(path, result.trace_id[:8])  # type: ignore[attr-defined]
    summary = stats(path)

    assert recent[0].attributes["agent.stop_reason"] == "completed"
    assert tree.splitlines()[1].startswith("· invoke_agent")
    assert "  · agent.turn" in tree
    assert "✗ execute_tool bash" in tree
    assert summary["runs"] == 1
    assert summary["tools"]["bash"] == {"calls": 2, "failure_rate": 1.0}
    assert summary["tools"]["read_file"]["failure_rate"] == 0.0


def test_render_tree_for_unknown_trace(tmp_path: Path) -> None:
    assert "no spans" in render_tree(tmp_path / "none.jsonl", "abc")


async def test_chat_span_carries_genai_usage_and_cost(spans: InMemorySpanExporter) -> None:
    from agent.llm import LLMClient
    from tests.llm.fakes import Recorder, chunk, model_config, sse_response, usage

    recorder = Recorder(
        sse_response(
            [chunk({"content": "hi"}), chunk(finish_reason="stop"), chunk(usage=usage(10, 5))]
        )
    )
    client = LLMClient(model_config(), http_client=recorder.transport())

    await client.complete([{"role": "user", "content": "hello"}])

    (span,) = [s for s in spans.get_finished_spans() if s.name == "chat fake-model"]
    a = span.attributes
    assert a is not None
    assert a["gen_ai.operation.name"] == "chat"
    assert a["gen_ai.request.model"] == "fake-model"
    assert a["gen_ai.usage.input_tokens"] == 10
    assert a["gen_ai.usage.output_tokens"] == 5
    assert a["gen_ai.response.finish_reasons"] == ("stop",)
    assert a["agent.model.revision"] == "rev-1"
    assert a["agent.cost_usd"] == (10 * 1.0 + 5 * 2.0) / 1e6
    assert a["server.address"] == "fake.test"


async def test_failed_chat_span_has_error_status(spans: InMemorySpanExporter) -> None:
    from agent.llm import LLMClient, LLMError
    from tests.llm.fakes import Recorder, error_response, model_config

    client = LLMClient(model_config(), http_client=Recorder(error_response(400)).transport())

    with pytest.raises(LLMError):
        await client.complete([{"role": "user", "content": "hello"}])

    (span,) = [s for s in spans.get_finished_spans() if s.name.startswith("chat")]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes is not None
    assert span.attributes["error.type"] == "bad_request"
