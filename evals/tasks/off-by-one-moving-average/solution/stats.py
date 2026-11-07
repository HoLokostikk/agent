"""Small statistics helpers."""


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty list")
    return sum(values) / len(values)


def moving_average(values: list[float], window: int) -> list[float]:
    """Mean of every run of `window` consecutive values."""
    if window <= 0:
        raise ValueError("window must be positive")
    return [mean(values[i : i + window]) for i in range(len(values) - window + 1)]
