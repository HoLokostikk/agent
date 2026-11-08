"""Assistant-only loss masking that works with any chat template.

For every message boundary the conversation prefix is rendered with the
tokenizer's own chat template; the token span a message adds is the
difference between consecutive prefix lengths. Only spans added by assistant
messages are trained on; system, user and tool tokens get label -100.

This is template-agnostic (no generation tags needed) and checkable by eye
with ``render_masking``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

IGNORE = -100


class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, Any]],
        *,
        tools: Sequence[dict[str, Any]] | None = ...,
        tokenize: bool = ...,
        add_generation_prompt: bool = ...,
    ) -> Any: ...

    def decode(self, ids: Sequence[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class Example:
    input_ids: list[int]
    labels: list[int]

    @property
    def trained_tokens(self) -> int:
        return sum(1 for label in self.labels if label != IGNORE)


class MaskingError(Exception):
    pass


def _ids(tokenizer: ChatTokenizer, messages: Sequence[dict[str, Any]], tools: Any) -> list[int]:
    out = tokenizer.apply_chat_template(
        list(messages), tools=tools, tokenize=True, add_generation_prompt=False
    )
    if isinstance(out, dict):
        out = out["input_ids"]
    return list(out)


def build_example(
    tokenizer: ChatTokenizer,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]] | None = None,
    max_length: int | None = None,
) -> Example:
    input_ids: list[int] = []
    labels: list[int] = []
    for i, message in enumerate(messages):
        ids = _ids(tokenizer, messages[: i + 1], tools)
        if ids[: len(input_ids)] != input_ids:
            raise MaskingError(f"chat template is not prefix-stable at message {i}")
        added = ids[len(input_ids) :]
        input_ids.extend(added)
        train = message["role"] == "assistant"
        labels.extend(added if train else [IGNORE] * len(added))
    if max_length is not None and len(input_ids) > max_length:
        raise MaskingError(f"example has {len(input_ids)} tokens > max_length {max_length}")
    return Example(input_ids, labels)


def render_masking(tokenizer: ChatTokenizer, example: Example, limit: int = 4000) -> str:
    """The conversation with trained spans in [[...]]: read it before training."""
    pieces: list[str] = []
    span: list[int] = []
    current: bool | None = None
    for token, label in zip(example.input_ids, example.labels, strict=True):
        trained = label != IGNORE
        if current is not None and trained != current:
            text = tokenizer.decode(span)
            pieces.append(f"[[{text}]]" if current else text)
            span = []
        current = trained
        span.append(token)
    if span:
        text = tokenizer.decode(span)
        pieces.append(f"[[{text}]]" if current else text)
    return "".join(pieces)[:limit]
