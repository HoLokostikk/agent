"""Error classification: which failures may be retried and which may not."""

from __future__ import annotations

from enum import StrEnum

import httpx2
import openai


class ErrorKind(StrEnum):
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    CONFLICT = "conflict"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    BAD_REQUEST = "bad_request"
    AUTH = "auth"
    NOT_FOUND = "not_found"
    STREAM_INTERRUPTED = "stream_interrupted"
    UNKNOWN = "unknown"


RETRYABLE: frozenset[ErrorKind] = frozenset(
    {
        ErrorKind.RATE_LIMIT,
        ErrorKind.SERVER,
        ErrorKind.CONFLICT,
        ErrorKind.TIMEOUT,
        ErrorKind.CONNECTION,
    }
)


class LLMError(Exception):
    """A classified model-call failure. Never carries request content."""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_s: float | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code
        self.retry_after_s = retry_after_s
        self.attempts = attempts

    @property
    def retryable(self) -> bool:
        return self.kind in RETRYABLE

    def __str__(self) -> str:
        status = f" HTTP {self.status_code}" if self.status_code is not None else ""
        return f"{self.kind}{status}: {self.message} (attempts: {self.attempts})"


def _kind_for_status(status: int) -> ErrorKind:
    if status == 429:
        return ErrorKind.RATE_LIMIT
    if status == 408:
        return ErrorKind.TIMEOUT
    if status == 409:
        return ErrorKind.CONFLICT
    if status >= 500:
        return ErrorKind.SERVER
    if status in (401, 403):
        return ErrorKind.AUTH
    if status == 404:
        return ErrorKind.NOT_FOUND
    if status in (400, 413, 422):
        return ErrorKind.BAD_REQUEST
    return ErrorKind.UNKNOWN


def parse_retry_after(headers: httpx2.Headers) -> float | None:
    """Seconds to wait, from ``retry-after-ms`` or numeric ``retry-after``.

    The HTTP-date form of ``retry-after`` is ignored: backoff applies instead.
    """
    for name, scale in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) / scale
        except ValueError:
            continue
        if seconds >= 0:
            return seconds
    return None


def classify(exc: BaseException) -> LLMError:
    """Map an SDK or transport exception onto an :class:`LLMError`."""
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, openai.APITimeoutError | httpx2.TimeoutException):
        return LLMError(ErrorKind.TIMEOUT, "request timed out")
    if isinstance(exc, openai.APIConnectionError | httpx2.TransportError):
        return LLMError(ErrorKind.CONNECTION, f"connection failed: {type(exc).__name__}")
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        return LLMError(
            _kind_for_status(status),
            exc.message,
            status_code=status,
            retry_after_s=parse_retry_after(exc.response.headers),
        )
    return LLMError(ErrorKind.UNKNOWN, f"{type(exc).__name__}: {exc}")
