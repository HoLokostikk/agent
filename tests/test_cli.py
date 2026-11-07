import contextlib
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from agent.cli import app as cli
from agent.llm import LLMClient, ModelConfig
from agent.sandbox import ExecResult
from tests.fakes import FakeLLM, FakeSandbox, call, say, use
from tests.llm.fakes import Recorder, chunk, error_response, sse_response, usage

runner = CliRunner()


def fake_factory(recorder: Recorder) -> object:
    def factory(config: ModelConfig) -> LLMClient:
        return LLMClient(config, http_client=recorder.transport())

    return factory


def test_models_lists_registry() -> None:
    result = runner.invoke(cli.app, ["models"])

    assert result.exit_code == 0
    assert "local-small" in result.output


def test_chat_prints_answer_and_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(
        sse_response(
            [
                chunk({"role": "assistant", "content": "pong"}),
                chunk(finish_reason="stop"),
                chunk(usage=usage(12, 3)),
            ]
        )
    )
    monkeypatch.setattr(cli, "make_client", fake_factory(recorder))

    result = runner.invoke(cli.app, ["chat", "-m", "local-small", "ping"])

    assert result.exit_code == 0
    assert "pong" in result.output
    assert "12 in / 3 out" in result.output
    assert "TTFT" in result.output


def test_chat_reports_model_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "make_client", fake_factory(Recorder(error_response(400))))

    result = runner.invoke(cli.app, ["chat", "-m", "local-small", "ping"])

    assert result.exit_code == 1
    assert "bad_request" in result.output


def test_chat_with_unknown_alias_is_a_config_error() -> None:
    result = runner.invoke(cli.app, ["chat", "-m", "nope", "ping"])

    assert result.exit_code == 2
    assert "unknown model alias" in result.output


def test_hosted_model_without_key_fails_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    result = runner.invoke(cli.app, ["chat", "-m", "qwen27b-hosted", "ping"])

    assert result.exit_code == 2
    assert "OPENROUTER_API_KEY" in result.output


# --- agent: headless mode ----------------------------------------------------


def install_fakes(monkeypatch: pytest.MonkeyPatch, llm: FakeLLM, sandbox: FakeSandbox) -> None:
    monkeypatch.setattr(cli, "make_client", lambda config: _Closable(llm, config))

    @contextlib.asynccontextmanager
    async def fake_sandbox(root: Path, kind: object) -> AsyncIterator[FakeSandbox]:
        yield sandbox

    monkeypatch.setattr(cli, "make_sandbox", fake_sandbox)


class _Closable:
    """FakeLLM plus the aclose() the CLI calls, keeping the real model config."""

    def __init__(self, llm: FakeLLM, config: ModelConfig) -> None:
        self._llm = llm
        self.config = config

    async def complete(self, *args: object, **kwargs: object) -> object:
        return await self._llm.complete(*args, **kwargs)  # type: ignore[arg-type]

    async def aclose(self) -> None:
        pass


def test_headless_json_reports_trajectory_and_stats(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    llm = FakeLLM(
        [
            use(call("edit_file", path="calc.py", old_string="a - b", new_string="a + b")),
            use(call("bash", command="pytest -q")),
            say("Fixed."),
        ]
    )
    sandbox = FakeSandbox(tmp_path, replies={"pytest -q": ExecResult(0, "1 passed")})
    install_fakes(monkeypatch, llm, sandbox)

    result = runner.invoke(
        cli.app, ["-p", "fix it", "--workspace", str(tmp_path), "--output", "json"]
    )

    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    assert data["stop_reason"] == "completed"
    assert data["final_answer"] == "Fixed."
    assert data["mode"] == "auto"  # headless default: no human to ask
    assert data["model"]["revision"].startswith("sha256:")
    assert data["prompt_version"] == "system.v1"
    assert data["stats"]["tool_calls"] == 2
    assert [c["name"] for t in data["turns"] for c in t["tool_calls"]] == ["edit_file", "bash"]
    assert data["messages"][0]["role"] == "system"
    assert (tmp_path / "calc.py").read_text().endswith("a + b\n")


def test_headless_exit_code_is_nonzero_when_not_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    llm = FakeLLM([use(call("list_dir", f"c{i}", path=str(i))) for i in range(5)])
    install_fakes(monkeypatch, llm, FakeSandbox(tmp_path))

    result = runner.invoke(
        cli.app,
        ["-p", "loop", "--workspace", str(tmp_path), "--output", "json", "--max-turns", "2"],
    )

    assert result.exit_code == 1
    assert json.loads(result.stdout)["stop_reason"] == "max_turns"


def test_headless_text_prints_answer_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fakes(monkeypatch, FakeLLM([say("All good.")]), FakeSandbox(tmp_path))

    result = runner.invoke(cli.app, ["-p", "check", "--workspace", str(tmp_path)])

    assert result.exit_code == 0
    assert "All good." in result.stdout
    assert "completed" in result.output


def test_missing_workspace_is_a_usage_error(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["-p", "x", "--workspace", str(tmp_path / "nope")])

    assert result.exit_code == 2


def test_eval_refuses_to_start_against_a_dead_endpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import httpx2

    monkeypatch.setattr(cli, "make_client", fake_factory(Recorder(httpx2.ConnectError("refused"))))

    result = runner.invoke(cli.app, ["eval", "toolcalls", "--out", str(tmp_path)])

    assert result.exit_code == 4
    assert "not usable" in result.output
    assert list(tmp_path.iterdir()) == []
