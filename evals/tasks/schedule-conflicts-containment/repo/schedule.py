from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    name: str
    start: int  # minutes from midnight
    end: int


def conflicts(events: list[Event]) -> set[tuple[str, str]]:
    """Pairs (a, b), a < b by name, of events whose time ranges overlap (end is exclusive)."""
    found: set[tuple[str, str]] = set()
    for i, a in enumerate(events):
        for b in events[i + 1 :]:
            if a.start <= b.start < a.end:
                found.add(tuple(sorted((a.name, b.name))))  # type: ignore[arg-type]
    return found
