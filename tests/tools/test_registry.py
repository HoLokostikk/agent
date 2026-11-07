from pathlib import Path

import pytest

from agent.sandbox import ExecResult
from agent.tools import Bash, PathGuard, ToolContext, ToolInputError, default_tools
from agent.tools.bash import BashArgs
from agent.tools.truncate import head_tail
from tests.fakes import FakeSandbox


def test_schemas_are_openai_tools_without_titles() -> None:
    schemas = default_tools().schemas()

    names = [s["function"]["name"] for s in schemas]
    assert names == ["read_file", "list_dir", "glob", "grep", "edit_file", "write_file", "bash"]
    for schema in schemas:
        assert "title" not in str(schema["function"].get("parameters"))
        assert schema["function"].get("description")


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("nope", "{}", "unknown tool"),
        ("read_file", "{not json", "not valid JSON"),
        ("read_file", "[1]", "must be a JSON object"),
        ("read_file", "{}", "path: Field required"),
        ("read_file", '{"path": "a", "limit": 0}', "limit"),
        ("read_file", '{"path": "a", "colour": 1}', "colour"),
    ],
)
def test_parse_errors_explain_the_problem(name: str, arguments: str, message: str) -> None:
    with pytest.raises(ToolInputError, match=message):
        default_tools().parse(name, arguments)


def test_empty_arguments_mean_empty_object() -> None:
    tool, args = default_tools().parse("list_dir", "")

    assert tool.name == "list_dir"
    assert args.model_dump() == {"path": "."}


async def test_bash_runs_in_sandbox_and_reports_exit_code(tmp_path: Path) -> None:
    sandbox = FakeSandbox(tmp_path, replies={"pytest -q": ExecResult(1, "1 failed")})
    ctx = ToolContext(PathGuard(tmp_path), sandbox)

    result = await Bash().run(BashArgs(command="pytest -q"), ctx)

    assert sandbox.commands == ["pytest -q"]
    assert not result.ok
    assert result.output == "exit code: 1\n1 failed"


async def test_bash_without_sandbox_refuses(tmp_path: Path) -> None:
    result = await Bash().run(BashArgs(command="ls"), ToolContext(PathGuard(tmp_path), None))

    assert not result.ok
    assert "no sandbox" in result.output


async def test_bash_reports_timeout(tmp_path: Path) -> None:
    sandbox = FakeSandbox(tmp_path, default=ExecResult(124, "", timed_out=True))

    result = await Bash().run(
        BashArgs(command="sleep 99", timeout_s=2), ToolContext(PathGuard(tmp_path), sandbox)
    )

    assert "timed out after 2s" in result.output


def test_head_tail_keeps_both_ends() -> None:
    text = "\n".join(f"line {i}" for i in range(1000))

    out = head_tail(text, max_lines=10)

    lines = out.splitlines()
    assert lines[:5] == [f"line {i}" for i in range(5)]
    assert lines[5] == "[... 990 lines truncated ...]"
    assert lines[-1] == "line 999"


def test_head_tail_caps_characters() -> None:
    out = head_tail("x" * 10_000, max_chars=100)

    assert len(out) < 200
    assert "characters truncated" in out
