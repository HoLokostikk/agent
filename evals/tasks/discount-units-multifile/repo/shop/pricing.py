def apply_discount(price: float, percent: float) -> float:
    """Price after a discount of `percent` percent (0-100)."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * (1 - percent / 100), 2)
