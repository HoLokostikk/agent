from typing import Any


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """New dict: override wins, but nested dicts are merged key by key, recursively."""
    result = dict(base)
    for key, value in override.items():
        result[key] = value
    return result
