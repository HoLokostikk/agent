import httpx2
import openai
import pytest

from agent.llm.errors import ErrorKind, classify, parse_retry_after

REQUEST = httpx2.Request("POST", "http://fake.test/v1/chat/completions")


def status_error(status: int, headers: dict[str, str] | None = None) -> openai.APIStatusError:
    response = httpx2.Response(status, headers=headers or {}, request=REQUEST)
    return openai.APIStatusError("boom", response=response, body=None)


@pytest.mark.parametrize(
    ("status", "kind", "retryable"),
    [
        (400, ErrorKind.BAD_REQUEST, False),
        (401, ErrorKind.AUTH, False),
        (404, ErrorKind.NOT_FOUND, False),
        (408, ErrorKind.TIMEOUT, True),
        (409, ErrorKind.CONFLICT, True),
        (422, ErrorKind.BAD_REQUEST, False),
        (429, ErrorKind.RATE_LIMIT, True),
        (500, ErrorKind.SERVER, True),
        (503, ErrorKind.SERVER, True),
        (418, ErrorKind.UNKNOWN, False),
    ],
)
def test_status_classification(status: int, kind: ErrorKind, retryable: bool) -> None:
    error = classify(status_error(status))

    assert error.kind is kind
    assert error.retryable is retryable
    assert error.status_code == status


def test_timeout_wins_over_connection() -> None:
    # APITimeoutError subclasses APIConnectionError.
    assert classify(openai.APITimeoutError(request=REQUEST)).kind is ErrorKind.TIMEOUT
    assert classify(openai.APIConnectionError(request=REQUEST)).kind is ErrorKind.CONNECTION


def test_unknown_exception_is_not_retryable() -> None:
    error = classify(RuntimeError("bug"))

    assert error.kind is ErrorKind.UNKNOWN
    assert not error.retryable


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "3"}, 3.0),
        ({"retry-after-ms": "250"}, 0.25),
        ({"retry-after-ms": "250", "retry-after": "9"}, 0.25),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({"retry-after": "-1"}, None),
        ({}, None),
    ],
)
def test_parse_retry_after(headers: dict[str, str], expected: float | None) -> None:
    assert parse_retry_after(httpx2.Headers(headers)) == expected
