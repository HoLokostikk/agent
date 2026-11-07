"""``search_code``: retrieval as a tool the agent calls like any other."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field

from agent.rag.search import Searcher
from agent.tools.base import Tool, ToolArgs, ToolContext, ToolKind, ToolResult

SNIPPET_LINES = 12


class SearchCodeArgs(ToolArgs):
    query: str = Field(description="What you are looking for, in words or identifiers.")
    k: int = Field(default=5, ge=1, le=20, description="Number of results.")


class SearchCode(Tool):
    name: ClassVar[str] = "search_code"
    description: ClassVar[str] = (
        "Semantic and keyword search over the indexed codebase. Returns the most "
        "relevant functions or blocks with file paths and line ranges. Use it when "
        "you do not know the exact name to grep for."
    )
    kind: ClassVar[ToolKind] = ToolKind.READ
    Args: ClassVar[type[ToolArgs]] = SearchCodeArgs

    def __init__(self, searcher: Searcher) -> None:
        self.searcher = searcher

    async def run(self, args: SearchCodeArgs, ctx: ToolContext) -> ToolResult:
        hits = await self.searcher.search(args.query, args.k)
        if not hits:
            return ToolResult(True, f"no results for {args.query!r}")
        blocks = []
        for hit in hits:
            c = hit.chunk
            lines = c.text.splitlines()
            snippet = "\n".join(lines[:SNIPPET_LINES])
            more = (
                f"\n  [... {len(lines) - SNIPPET_LINES} more lines]"
                if len(lines) > SNIPPET_LINES
                else ""
            )
            blocks.append(f"{c.path}:{c.start_line}-{c.end_line} ({c.symbol})\n{snippet}{more}")
        return ToolResult(True, "\n\n".join(blocks))
