from store.catalog import PRODUCTS
from store.inventory import restock


def test_restock_by_sku() -> None:
    assert restock("SKU-2", 5) == 5
    assert PRODUCTS["SKU-2"]["stock"] == 5
