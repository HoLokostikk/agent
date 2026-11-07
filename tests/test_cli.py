import pytest
from typer.testing import CliRunner

from agent.cli import app as cli
from agent.llm import LLMClient, ModelConfig
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
