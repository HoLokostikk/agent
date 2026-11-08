"""MCP server exposing the code index: ``python -m agent.mcp.server --workspace PATH``."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from agent.llm.config import get_model
from agent.rag.embed import OpenAIEmbedder
from agent.rag.index import CodeIndex
from agent.rag.search import Searcher


def build_server(searcher: Searcher, stats: dict[str, object]) -> MCPServer:
    server = MCPServer(
        name="code-search",
        instructions="Search the indexed repository by meaning or identifiers.",
    )

    @server.tool(description="Search the codebase. Returns path:start-end (symbol) and a snippet.")
    async def search_code(query: str, k: int = 5) -> str:
        hits = await searcher.search(query, max(1, min(k, 20)))
        if not hits:
            return f"no results for {query!r}"
        return "\n\n".join(
            f"{h.chunk.header()}\n" + "\n".join(h.chunk.text.splitlines()[:12]) for h in hits
        )

    @server.tool(description="Find a function, class or method by name and return its full source.")
    def get_symbol(name: str) -> str:
        matches = [
            c for c in searcher.chunks if c.symbol.split()[-1].split(".")[-1] == name.split(".")[-1]
        ]
        if not matches:
            return f"symbol {name!r} not found"
        return "\n\n".join(f"{c.header()}\n{c.text}" for c in matches[:5])

    @server.resource("index://stats", description="Index statistics.", mime_type="application/json")
    def index_stats() -> str:
        return json.dumps(stats)

    return server


async def _prepare(
    workspace: Path, config: Path, embed_model: str | None
) -> tuple[Searcher, dict[str, object]]:
    embedder = OpenAIEmbedder(get_model(config, embed_model)) if embed_model else None
    index = CodeIndex(workspace)
    try:
        update = await index.update(embedder)
        chunks, vectors = index.vectors()
    finally:
        index.close()
    stats: dict[str, object] = {
        "workspace": workspace.name,
        "files": update.files_seen,
        "chunks": update.chunks_total,
        "embedder": embedder.name if embedder else None,
    }
    return Searcher(chunks, vectors, embedder), stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path())
    parser.add_argument("--config", type=Path, default=Path("configs/models.yaml"))
    parser.add_argument("--embed-model")
    args = parser.parse_args()
    searcher, stats = asyncio.run(_prepare(args.workspace.resolve(), args.config, args.embed_model))
    build_server(searcher, stats).run("stdio")


if __name__ == "__main__":
    main()
