import functools
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry(times: int) -> Callable[[Callable[[], T]], Callable[[], T]]:
    """Call the function up to `times` times in total; re-raise the last error."""

    def wrap(fn: Callable[[], T]) -> Callable[[], T]:
        @functools.wraps(fn)
        def inner() -> T:
            last: Exception | None = None
            for _ in range(times + 1):
                try:
                    return fn()
                except Exception as exc:
                    last = exc
            assert last is not None
            raise last

        return inner

    return wrap
