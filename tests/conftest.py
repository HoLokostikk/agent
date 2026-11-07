import pytest


@pytest.fixture(autouse=True)
def _no_trace_files(monkeypatch: pytest.MonkeyPatch) -> None:
    # CLI tests must not write spans into the developer's ~/.local/state.
    monkeypatch.setenv("AGENT_TRACE", "none")
