from store.catalog import PRODUCTS


def restock(sku: str, amount: int) -> int:
    """Add stock to a product by SKU and return the new stock level."""
    product = next(p for p in PRODUCTS.values() if p["name"] == sku)
    product["stock"] += amount
    return int(product["stock"])
