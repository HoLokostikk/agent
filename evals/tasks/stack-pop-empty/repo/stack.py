class Stack:
    def __init__(self) -> None:
        self._items: list[int] = []

    def push(self, item: int) -> None:
        self._items.append(item)

    def pop(self) -> int | None:
        """Remove and return the top item; IndexError("pop from empty stack") if empty."""
        if not self._items:
            return None
        return self._items.pop()
