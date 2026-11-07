"""Smoke test: the package and every subpackage import cleanly."""

import importlib

import pytest

SUBPACKAGES = [
    "agent",
    "agent.cli",
    "agent.llm",
    "agent.core",
    "agent.tools",
    "agent.sandbox",
    "agent.permissions",
    "agent.rag",
    "agent.mcp",
    "agent.telemetry",
]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(name)
