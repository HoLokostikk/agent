import pytest


@pytest.fixture(autouse=True)
def _isolated_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> None:
    monkeypatch.setenv("AGENT_TRACE", "none")
    monkeypatch.setenv("AGENT_STATE_DIR", str(tmp_path_factory.mktemp("state")))
