"""Redaction of secrets and personal data before anything leaves the process.

Applied to every span attribute (see ``telemetry.tracing.set_attributes``).
Patterns are deliberately broad: a false positive costs a less useful trace,
a false negative leaks a credential into a log that may be shipped elsewhere.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

_RULES: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL
        ),
        "[REDACTED_PRIVATE_KEY]",
    ),
    (re.compile(r"\bsk-(?:or-v1-|ant-|proj-)?[A-Za-z0-9_\-]{16,}"), "[REDACTED_API_KEY]"),
    (
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b|\bgithub_pat_[A-Za-z0-9_]{20,}"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"), "[REDACTED_HF_TOKEN]"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer [REDACTED]"),
    (
        re.compile(
            r"(?im)^(\s*(?:export\s+)?[A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD|PASSWD)[A-Z0-9_]*"
            r"\s*[=:]\s*)\S+"
        ),
        r"\1[REDACTED]",
    ),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
]


def redact(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def redact_value(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return [redact(v) if isinstance(v, str) else v for v in value]
    return value
