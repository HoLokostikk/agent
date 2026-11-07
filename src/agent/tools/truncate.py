"""Keep tool output short: head + tail with an explicit marker in between.

The head usually holds the command and the first error; the tail holds the
summary line (``3 failed, 10 passed``). The middle is what gets dropped.
"""

from __future__ import annotations


def head_tail(text: str, *, max_lines: int = 120, max_chars: int = 6000) -> str:
    lines = text.splitlines()
    if len(lines) > max_lines:
        head_n = max_lines // 2
        tail_n = max_lines - head_n
        dropped = len(lines) - max_lines
        lines = [*lines[:head_n], f"[... {dropped} lines truncated ...]", *lines[-tail_n:]]
    out = "\n".join(lines)
    if len(out) > max_chars:
        half = max_chars // 2
        dropped_chars = len(out) - max_chars
        out = f"{out[:half]}\n[... {dropped_chars} characters truncated ...]\n{out[-half:]}"
    return out
