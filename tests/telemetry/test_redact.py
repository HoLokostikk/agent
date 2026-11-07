import pytest

from agent.telemetry import redact, redact_value


@pytest.mark.parametrize(
    ("text", "leaked"),
    [
        ("key sk-or-v1-0123456789abcdef0123456789abcdef", "0123456789abcdef0123"),
        ("OPENAI sk-proj-AbCdEfGhIjKlMnOpQrStUv", "AbCdEfGhIjKlMnOp"),
        ("token ghp_0123456789abcdefghijABCDEFGHIJ", "0123456789abcdefghij"),
        ("hf_abcdefghijklmnopqrstuvwxyz12", "abcdefghijklmnopqrst"),
        ("aws AKIAIOSFODNN7EXAMPLE", "IOSFODNN7EXAMPLE"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9"),
        ("DATABASE_PASSWORD=hunter2hunter2", "hunter2"),
        ("export STRIPE_SECRET_KEY=whsec_live_abc", "whsec_live_abc"),
        ("mail me at jane.doe@example.com", "jane.doe@example.com"),
        (
            "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\n-----END RSA PRIVATE KEY-----",
            "MIIEowIBAAKCAQEA",
        ),
    ],
)
def test_secrets_are_redacted(text: str, leaked: str) -> None:
    assert leaked not in redact(text)
    assert "REDACTED" in redact(text)


@pytest.mark.parametrize(
    "text", ["python -m pytest -q", "exit code: 1", "def tokenize(text): ...", "sk-short"]
)
def test_ordinary_text_is_untouched(text: str) -> None:
    assert redact(text) == text


def test_redact_value_handles_lists_and_non_strings() -> None:
    assert redact_value(["a", "sk-ant-0123456789abcdefghij"]) == ["a", "[REDACTED_API_KEY]"]
    assert redact_value(42) == 42
