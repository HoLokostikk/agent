from datetime import date


def days_inclusive(start: date, end: date) -> int:
    """Number of calendar days from start to end, counting both ends."""
    return (end - start).days + 1
