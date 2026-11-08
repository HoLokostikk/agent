"""Against a real code-search MCP server over stdio."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from agent.mcp.client import (
    MCPArgs,
    MCPConfig,
    MCPHub,
    PinStore,
    ServerConfig,
    load_mcp_config,
    tool_fingerprint,
)
from agent.mcp.server import build_server
from agent.rag.chunker import chunk_file
from agent.rag.search import Searcher
from agent.tools import PathGuard, ToolContext, ToolKind


def server_config(workspace: Path, **kw: object) -> ServerConfig:
    return ServerConfig(
        command=sys.executable,
        args=["-m", "agent.mcp.server", "--workspace", str(workspace)],
        kind="read",
        timeout_s=60,
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "billing.py").write_text("def compute_invoice_total(items):\n    return sum(items)\n")
    return ws


async def test_server_tools_become_namespaced_agent_tools(workspace: Path, tmp_path: Path) -> None:
    pins = PinStore(tmp_path / "pins.json")
    async with MCPHub(MCPConfig(servers={"cs": server_config(workspace)}), pins=pins) as hub:
        names = sorted(t.name for t in hub.tools)
        search = next(t for t in hub.tools if t.name == "mcp__cs__search_code")
        result = await search.run(
            MCPArgs.model_validate({"query": "invoice total"}),
            ToolContext(PathGuard(workspace), None),
        )

    assert names == ["mcp__cs__get_symbol", "mcp__cs__search_code"]
    assert search.kind is ToolKind.READ
    params: Any = search.schema()["function"].get("parameters")
    assert "query" in params["properties"]
    assert result.ok
    assert result.output.startswith('<untrusted-tool-output source="mcp:cs">')
    assert "billing.py:1-2" in result.output
    assert any("trusted on first use" in n for n in hub.notices)
    assert set(json.loads((tmp_path / "pins.json").read_text())["cs"]) == {
        "search_code",
        "get_symbol",
    }


async def test_changed_description_disables_the_tool(workspace: Path, tmp_path: Path) -> None:
    pins = PinStore(tmp_path / "pins.json")
    pins.pin("cs", "search_code", tool_fingerprint("search_code", "an older description", {}))
    pins.save()

    async with MCPHub(
        MCPConfig(servers={"cs": server_config(workspace)}), pins=PinStore(tmp_path / "pins.json")
    ) as hub:
        names = [t.name for t in hub.tools]

    assert "mcp__cs__search_code" not in names
    assert "mcp__cs__get_symbol" in names
    assert any("changed its description" in n for n in hub.notices)


async def test_allowlist_and_unavailable_server(workspace: Path, tmp_path: Path) -> None:
    config = MCPConfig(
        servers={
            "cs": server_config(workspace, allow=["get_symbol"]),
            "broken": ServerConfig(command="/nonexistent/binary", timeout_s=5),
        }
    )
    async with MCPHub(config, pins=PinStore(tmp_path / "pins.json")) as hub:
        names = [t.name for t in hub.tools]

    assert names == ["mcp__cs__get_symbol"]
    assert any("broken unavailable" in n for n in hub.notices)


async def test_server_functions_directly() -> None:
    searcher = Searcher(
        chunk_file("a.py", "def alpha():\n    return 1\n\n\nclass Beta:\n    pass\n")
    )
    server = build_server(searcher, {"files": 1})

    tools = {t.name for t in await server.list_tools()}

    assert tools == {"search_code", "get_symbol"}


def test_repo_mcp_config_is_valid() -> None:
    config = load_mcp_config(Path(__file__).parents[2] / "configs" / "mcp.json")

    assert config.servers["code-search"].kind == "read"
