import pytest

from retry import retry


def test_succeeds_after_failures() -> None:
    calls = []

    @retry(times=3)
    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ValueError("not yet")
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3


def test_total_attempts_and_reraise() -> None:
    calls = []

    @retry(times=3)
    def always() -> None:
        calls.append(1)
        raise KeyError("boom")

    with pytest.raises(KeyError):
        always()
    assert len(calls) == 3
