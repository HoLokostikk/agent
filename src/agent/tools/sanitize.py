"""Strip terminal control sequences from untrusted text.

A file or command output can carry ANSI/OSC escapes that rewrite what the
terminal shows (hide a line, fake a prompt, set the window title, even write
to the clipboard via OSC 52). Neither the user's terminal nor the model has
any use for them.
"""

from __future__ import annotations

import re

_ESCAPES = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"
    r"|\x1b[PX^_][^\x1b]*\x1b\\"
    r"|\x1b[@-Z\\-_]"
    r"|\x9b[0-?]*[ -/]*[@-~]"
)
_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Remove escape sequences and control characters; keep \\n, \\t and \\r\\n."""
    text = _ESCAPES.sub("", text)
    text = text.replace("\r\n", "\n")
    return _CONTROLS.sub("", text.replace("\r", ""))
