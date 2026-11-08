from pathlib import Path

import pytest

from agent.llm.config import ConfigError, get_model, load_models

REGISTRY = Path(__file__).parents[2] / "configs" / "models.yaml"


def test_repo_registry_loads_and_every_model_is_pinned() -> None:
    models = load_models(REGISTRY)

    assert {"local-small", "qwen27b-hosted"} <= set(models)
    for alias, model in models.items():
        assert model.alias == alias
        assert model.revision.strip()
        if "localhost:11434" in model.base_url:
            assert model.revision.startswith("sha256:")


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  m:\n    base_url: x\n    model: y\n    revision: r\n"
        "    context_window: 1\n    pricing: {input_per_mtok: 0, output_per_mtok: 0}\n"
        "    temperatur: 0.5\n"
    )

    with pytest.raises(ConfigError, match="temperatur"):
        load_models(path)


def test_missing_revision_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  m:\n    base_url: x\n    model: y\n    revision: ''\n"
        "    context_window: 1\n    pricing: {input_per_mtok: 0, output_per_mtok: 0}\n"
    )

    with pytest.raises(ConfigError, match="revision"):
        load_models(path)


def test_unknown_alias_lists_known_ones() -> None:
    with pytest.raises(ConfigError, match="local-small"):
        get_model(REGISTRY, "nope")


def test_api_key_comes_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    model = get_model(REGISTRY, "qwen27b-hosted")

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        model.resolve_api_key()

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert model.resolve_api_key() == "sk-test"


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_models(tmp_path / "absent.yaml")
