import pytest


@pytest.fixture(autouse=True)
def _isolated_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    # CLI tests must not write spans or sessions into the developer's ~/.local/state.
    monkeypatch.setenv("AGENT_TRACE", "none")
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path_factory.mktemp("state")))
