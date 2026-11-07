"""Context budget and compaction.

The server's ``usage.input_tokens`` from the last call is exact; messages
appended since are estimated at ~3 characters per token (conservative for
code). When the estimate passes ``compact_fraction`` of the window, old turns
are replaced by a summary written by the (possibly cheaper) compaction model.

What survives a compaction: the system prompt, the current task verbatim, the
summary, and the most recent messages, cut only where no tool result would be
separated from the assistant message that requested it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from openai.types.chat import ChatCompletionMessageParam

if TYPE_CHECKING:
    from agent.core.loop import ChatModel

CHARS_PER_TOKEN = 3.0

SUMMARY_PROMPT = """You are compressing the history of a coding agent's session so it can continue
with less context. Write a concise summary for the agent itself, with these sections:

Task: the user's goal in one or two sentences.
Done so far: actions taken and their results, including files read and changed.
Findings: facts learned about the code (file paths, function names, error messages).
Open problems: what still fails or is unknown.
Next step: the single most useful next action.

Keep exact file paths, identifiers and error messages. Do not invent anything."""

SUMMARY_HEADER = "Summary of the earlier part of this session (older turns were compacted):\n\n"


def estimate_tokens(messages: Sequence[ChatCompletionMessageParam]) -> int:
    chars = sum(len(json.dumps(m, ensure_ascii=False, default=str)) for m in messages)
    return int(chars / CHARS_PER_TOKEN)


def safe_cut(
    messages: Sequence[ChatCompletionMessageParam], keep_last: int, floor: int
) -> int | None:
    """Index where the kept tail starts, or None if nothing can be dropped.

    The tail must not start with a ``tool`` message: its assistant message
    (with the matching ``tool_calls``) would be gone, which servers reject.
    """
    start = max(floor, len(messages) - keep_last)
    while start < len(messages) and messages[start]["role"] == "tool":
        start += 1
    if start >= len(messages) or start <= floor:
        return None
    return start


def _render_for_summary(messages: Sequence[ChatCompletionMessageParam], limit: int = 24_000) -> str:
    parts = []
    for m in messages:
        role = m["role"]
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content, default=str)
        calls = m.get("tool_calls")
        if calls:
            text = (text or "") + "\n" + json.dumps(calls, default=str)
        parts.append(f"[{role}] {text}")
    joined = "\n\n".join(parts)
    if len(joined) > limit:
        joined = joined[: limit // 2] + "\n[... middle omitted ...]\n" + joined[-limit // 2 :]
    return joined


@dataclass
class Compactor:
    model: ChatModel
    keep_last: int = 6
    compact_fraction: float = 0.7

    def should_compact(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        last_input_tokens: int | None,
        messages_at_last_call: int,
        window: int,
    ) -> bool:
        if last_input_tokens is None:
            projected = estimate_tokens(messages)
        else:
            projected = last_input_tokens + estimate_tokens(messages[messages_at_last_call:])
        return projected >= window * self.compact_fraction

    async def compact(
        self, messages: list[ChatCompletionMessageParam], task_index: int
    ) -> list[ChatCompletionMessageParam] | None:
        """Return the compacted history, or None if there is nothing to drop.

        ``messages[0]`` is the system prompt and ``messages[task_index]`` the
        current task; both are kept verbatim.
        """
        cut = safe_cut(messages, self.keep_last, floor=task_index + 1)
        if cut is None:
            return None
        dropped = messages[1:cut]
        dropped = [m for i, m in enumerate(dropped, 1) if i != task_index]
        completion = await self.model.complete(
            [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": _render_for_summary(dropped)},
            ],
            tools=None,
        )
        summary = completion.content.strip() or "(the summary came back empty)"
        return [
            messages[0],
            messages[task_index],
            {"role": "user", "content": SUMMARY_HEADER + summary},
            *messages[cut:],
        ]
