from collections.abc import Sequence
from typing import Any

import pytest

from agent.training.masking import IGNORE, MaskingError, build_example, render_masking


class CharTokenizer:
    """ChatML-like template over characters: one token per character."""

    def render(self, messages: Sequence[dict[str, Any]]) -> str:
        out = ""
        for m in messages:
            body = m.get("content") or ""
            for call in m.get("tool_calls") or []:
                body += f"<call {call['function']['name']}>"
            out += f"<{m['role']}>{body}</>"
        return out

    def apply_chat_template(
        self,
        conversation: Any,
        *,
        tools: Any = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> Any:
        return [ord(c) for c in self.render(conversation)]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(i) for i in ids)


MESSAGES: list[dict[str, Any]] = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "fix"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}],
    },
    {"role": "tool", "content": "SECRET OUTPUT"},
    {"role": "assistant", "content": "done"},
]


def test_only_assistant_tokens_are_trained() -> None:
    tok = CharTokenizer()

    example = build_example(tok, MESSAGES)
    shown = render_masking(tok, example)

    assert shown == (
        "<system>sys</><user>fix</>[[<assistant><call read_file></>]]"
        "<tool>SECRET OUTPUT</>[[<assistant>done</>]]"
    )
    trained = "".join(
        chr(t)
        for t, label in zip(example.input_ids, example.labels, strict=True)
        if label != IGNORE
    )
    assert "SECRET" not in trained
    assert example.trained_tokens == len("<assistant><call read_file></>") + len(
        "<assistant>done</>"
    )


def test_non_prefix_stable_template_is_rejected() -> None:
    class Dropping(CharTokenizer):
        def render(self, messages: Sequence[dict[str, Any]]) -> str:
            # Re-renders history differently once a tool message appears.
            text = super().render(messages)
            return (
                text.replace("<user>", "<USER>")
                if any(m["role"] == "tool" for m in messages)
                else text
            )

    with pytest.raises(MaskingError, match="prefix-stable"):
        build_example(Dropping(), MESSAGES)


def test_max_length() -> None:
    with pytest.raises(MaskingError, match="max_length"):
        build_example(CharTokenizer(), MESSAGES, max_length=10)
