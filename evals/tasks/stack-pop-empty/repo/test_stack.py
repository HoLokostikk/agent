import pytest

from stack import Stack


def test_push_pop() -> None:
    s = Stack()
    s.push(1)
    s.push(2)
    assert s.pop() == 2


def test_pop_empty_raises() -> None:
    with pytest.raises(IndexError, match="pop from empty stack"):
        Stack().pop()
