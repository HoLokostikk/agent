def safe_divide(a: float, b: float) -> float | None:
    """Return a / b, or None when b is zero."""
    if b == 0:
        return None
    return a / b
