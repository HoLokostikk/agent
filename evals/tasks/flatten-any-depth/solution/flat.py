from typing import Any


def flatten(items: list[Any]) -> list[Any]:
    """Flatten nested lists of any depth, keeping order."""
    out: list[Any] = []
    for item in items:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out
