from collections import OrderedDict


class LRUCache:
    """Keeps at most `capacity` items; evicts the least recently used one."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.items: OrderedDict[str, int] = OrderedDict()

    def get(self, key: str) -> int | None:
        if key not in self.items:
            return None
        self.items.move_to_end(key)
        return self.items[key]

    def put(self, key: str, value: int) -> None:
        self.items[key] = value
        self.items.move_to_end(key)
        if len(self.items) > self.capacity:
            self.items.popitem(last=True)
