from pathlib import Path
from typing import Any

import pytest
from openai.types.chat import ChatCompletionMessageParam

from agent.core import Agent, Budget, Compactor, StopReason, estimate_tokens
from agent.core.builder import AgentSpec, build_agent
from agent.core.context import SUMMARY_HEADER, safe_cut
from agent.core.session import SessionStore, build_system_prompt
from agent.permissions import Mode, PermissionPolicy, load_rules
from agent.tools import PathGuard, ToolContext, default_tools
from tests.fakes import FakeLLM, call, fake_config, say, use

RULES = load_rules(Path(__file__).parents[2] / "configs" / "permissions.yaml")


def history() -> list[ChatCompletionMessageParam]:
    msgs: list[Any] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "the task"},
    ]
    for i in range(4):
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {"name": "list_dir", "arguments": "{}"},
                    }
                ],
            }
        )
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "x" * 300})
    return msgs


def test_safe_cut_never_starts_the_tail_with_a_tool_result() -> None:
    msgs = history()

    cut = safe_cut(msgs, keep_last=3, floor=2)

    assert cut is not None
    assert msgs[cut]["role"] == "assistant"


def test_safe_cut_returns_none_when_nothing_can_go() -> None:
    assert safe_cut(history()[:4], keep_last=6, floor=2) is None


async def test_compaction_keeps_system_task_summary_and_tail() -> None:
    summarizer = FakeLLM([say("Task: x. Done so far: listed dirs.")])
    compactor = Compactor(summarizer, keep_last=2)
    msgs = history()

    compacted = await compactor.compact(msgs, task_index=1)

    assert compacted is not None
    assert [m["role"] for m in compacted[:3]] == ["system", "user", "user"]
    assert compacted[1]["content"] == "the task"
    assert str(compacted[2]["content"]).startswith(SUMMARY_HEADER)
    assert compacted[3]["role"] == "assistant"
    assert compacted[-1]["role"] == "tool"
    sent = str(summarizer.seen[0][1]["content"])
    assert "list_dir" in sent
    assert "sys" not in sent.split("[")[0]


def test_should_compact_uses_real_usage_plus_estimate() -> None:
    compactor = Compactor(FakeLLM([]), compact_fraction=0.7)
    msgs = history()

    assert not compactor.should_compact(
        msgs, last_input_tokens=100, messages_at_last_call=len(msgs), window=1000
    )
    assert compactor.should_compact(
        msgs, last_input_tokens=750, messages_at_last_call=len(msgs), window=1000
    )
    assert compactor.should_compact(
        msgs, last_input_tokens=None, messages_at_last_call=0, window=estimate_tokens(msgs)
    )


async def test_agent_compacts_instead_of_stopping(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    steps = [
        use(call("read_file", f"c{i}", path="a.py", offset=1, limit=i + 1), input_tokens=650)
        for i in range(5)
    ]
    llm = FakeLLM([*steps, say("done")], config=fake_config(context_window=1000))
    summarizer = FakeLLM([say(f"summary {i}") for i in range(10)])
    agent = Agent(
        llm,
        default_tools(),
        ToolContext(PathGuard(tmp_path), None),
        PermissionPolicy(Mode.AUTO, RULES),
        system_prompt="s",
        budget=Budget(max_context_fraction=0.95),
        compactor=Compactor(summarizer, keep_last=2, compact_fraction=0.6),
    )

    result = await agent.run("read a lot")

    assert result.stop_reason is StopReason.COMPLETED
    assert result.stats.compactions >= 1
    assert agent.messages[1]["content"] == "read a lot"


def test_cache_hit_rate_is_reported() -> None:
    from agent.core.loop import RunStats

    stats = RunStats(input_tokens=1000, cached_input_tokens=800)

    assert stats.cache_hit_rate == 0.8
    assert stats.as_dict()["cache_hit_rate"] == 0.8


def test_project_instructions_are_appended(tmp_path: Path) -> None:
    assert build_system_prompt("base", tmp_path) == "base"
    (tmp_path / "AGENTS.md").write_text("Use tabs. Run make test.")

    prompt = build_system_prompt("base", tmp_path)

    assert prompt.startswith("base")
    assert "Project instructions from AGENTS.md" in prompt
    assert "Run make test." in prompt


def test_session_round_trip_and_last(tmp_path: Path) -> None:
    store = SessionStore(tmp_path)
    msgs: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "hi"},
    ]

    store.save("one", msgs, {"model": "m", "workspace": "/w"})
    meta, loaded = store.load("last")

    assert meta["id"] == "one"
    assert loaded == msgs
    assert store.list()[0].messages == 2


def test_missing_last_session_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SessionStore(tmp_path).load("last")


def spec(**kw: Any) -> AgentSpec:
    return AgentSpec(
        base_prompt="b", prompt_version="v", rules=RULES, mode=Mode.AUTO, budget=Budget(), **kw
    )


def test_extended_tools_depend_on_context_window(tmp_path: Path) -> None:
    small = build_agent(FakeLLM([], config=fake_config(8192)), tmp_path, None, spec())
    big = build_agent(FakeLLM([], config=fake_config(262_144)), tmp_path, None, spec())

    assert "explore" not in small.tools.names
    assert {"todo_write", "explore"} <= set(big.tools.names)


async def test_todo_tool_renders_plan(tmp_path: Path) -> None:
    agent = build_agent(
        FakeLLM(
            [
                use(
                    call(
                        "todo_write",
                        items=[
                            {"content": "read", "status": "done"},
                            {"content": "fix", "status": "in_progress"},
                        ],
                    )
                ),
                say("ok"),
            ],
            config=fake_config(100_000),
        ),
        tmp_path,
        None,
        spec(),
    )

    result = await agent.run("plan")

    output = result.turns[0].tool_calls[0].output
    assert output.splitlines() == ["plan (1/2 done):", "[x] read", "[~] fix"]


async def test_explorer_runs_in_its_own_context(tmp_path: Path) -> None:
    (tmp_path / "auth.py").write_text("def login(): ...\n")
    explorer_model = FakeLLM(
        [use(call("grep", pattern="def login")), say("login is defined in auth.py:1")],
        config=fake_config(100_000),
    )
    main = FakeLLM(
        [use(call("explore", question="where is login?")), say("found it")],
        config=fake_config(100_000),
    )
    agent = build_agent(main, tmp_path, None, spec(), compact_model=explorer_model)

    result = await agent.run("find login")

    output = result.turns[0].tool_calls[0].output
    assert output.startswith("login is defined in auth.py:1")
    assert "[explorer: 2 turns, 1 tool calls, completed]" in output
    assert all("def login" not in str(m.get("content")) for m in agent.messages)
