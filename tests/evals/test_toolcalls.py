from pathlib import Path

import pytest

from agent.evals.toolcalls import (
    Case,
    Expectation,
    grade,
    load_cases,
    match_value,
    run_cases,
    summarize_cases,
)
from agent.tools import default_tools
from tests.fakes import FakeLLM, call, raw_call, say, use

CASES = Path(__file__).parents[2] / "evals" / "tool_calls" / "cases.yaml"


@pytest.mark.parametrize(
    ("expected", "actual", "ok"),
    [
        ("a.py", "a.py", True),
        ("a.py", " a.py ", True),
        ({"endswith": "a.py"}, "src/a.py", True),
        ({"endswith": "a.py"}, "a.pyc", False),
        ({"contains": "test_"}, "**/test_*.py", True),
        ({"regex": r"pytest.*-q"}, "python -m pytest -q", True),
        ({"any_of": [".", ""]}, "", True),
        ({"any_of": [".", ""]}, "src", False),
        (20, 20, True),
        (20, "20", False),
        (True, True, True),
        ({"contains": "x"}, None, False),
    ],
)
def test_matchers(expected: object, actual: object, ok: bool) -> None:
    assert match_value(expected, actual) is ok


CASE = Case(
    id="c",
    prompt="p",
    expect=[
        Expectation(tool="read_file", args={"path": {"endswith": "a.py"}}),
        Expectation(tool="grep"),
    ],
)


@pytest.mark.parametrize(
    ("tool", "arguments", "outcome"),
    [
        ("read_file", '{"path": "src/a.py"}', "correct"),
        ("grep", '{"pattern": "x"}', "correct"),
        ("read_file", '{"path": "b.py"}', "wrong_args"),
        ("bash", '{"command": "cat a.py"}', "wrong_tool"),
        ("read_file", '{"path": ', "invalid_json"),
        (None, None, "no_call"),
    ],
)
def test_grade(tool: str | None, arguments: str | None, outcome: str) -> None:
    assert grade(CASE, tool, arguments)[0] == outcome


def test_repo_cases_load_and_reference_real_tools() -> None:
    cases = load_cases(CASES)
    tools = set(default_tools().names)

    assert len(cases) >= 30
    for case in cases:
        assert {e.tool for e in case.expect} <= tools, case.id


async def test_run_and_summarize() -> None:
    cases = [
        Case(id="a", prompt="read a.py", expect=[Expectation(tool="read_file")]),
        Case(id="b", prompt="run tests", expect=[Expectation(tool="bash")]),
        Case(id="c", prompt="grep x", expect=[Expectation(tool="grep")]),
    ]
    llm = FakeLLM(
        [
            use(call("read_file", path="a.py")),
            say("I would run pytest"),
            use(raw_call("grep", "{bad")),
        ]
    )

    results = await run_cases(cases, llm, default_tools(), "system")
    summary = summarize_cases(results)

    assert [r.outcome for r in results] == ["correct", "no_call", "invalid_json"]
    assert summary["accuracy"]["estimate"] == pytest.approx(1 / 3)
    assert summary["invalid_json_rate"] == pytest.approx(1 / 2)
    assert summary["no_call_rate"] == pytest.approx(1 / 3)
    assert {f["case_id"] for f in summary["failures"]} == {"b", "c"}


def test_omitted_arguments_are_graded_as_their_defaults() -> None:
    case = Case(id="root", prompt="p", expect=[Expectation(tool="list_dir", args={"path": "."})])

    assert grade(case, "list_dir", "{}") == ("wrong_args", "mismatched: path")
    assert grade(case, "list_dir", "{}", default_tools())[0] == "correct"
    assert grade(case, "list_dir", '{"bogus": 1}', default_tools())[0] == "wrong_args"
