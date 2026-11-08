from typing import Any


def flatten(items: list[Any]) -> list[Any]:
    """Flatten nested lists of any depth, keeping order."""
    out: list[Any] = []
    for item in items:
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out
