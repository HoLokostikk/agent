"""Strip terminal control sequences from untrusted text.

A file or command output can carry ANSI/OSC escapes that rewrite what the
terminal shows (hide a line, fake a prompt, set the window title, even write
to the clipboard via OSC 52). Neither the user's terminal nor the model has
any use for them.
"""

from __future__ import annotations

import re

# CSI ... final byte, OSC ... BEL/ST, other ESC sequences, then C0/C1 controls.
_ESCAPES = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC
    r"|\x1b[PX^_][^\x1b]*\x1b\\"  # DCS, SOS, PM, APC
    r"|\x1b[@-Z\\-_]"  # two-byte escapes
    r"|\x9b[0-?]*[ -/]*[@-~]"  # 8-bit CSI
)
_CONTROLS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")


def sanitize(text: str) -> str:
    """Remove escape sequences and control characters; keep \\n, \\t and \\r\\n."""
    text = _ESCAPES.sub("", text)
    text = text.replace("\r\n", "\n")
    return _CONTROLS.sub("", text.replace("\r", ""))
