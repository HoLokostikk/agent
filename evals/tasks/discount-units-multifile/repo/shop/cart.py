from shop.pricing import apply_discount

MEMBER_DISCOUNT = 0.1  # ten percent off for members


def checkout(prices: list[float], member: bool = False) -> float:
    total = sum(prices)
    if member:
        return apply_discount(total, MEMBER_DISCOUNT)
    return round(total, 2)
