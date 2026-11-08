import pytest

from bank.account import Account, InsufficientFunds
from bank.transfer import transfer


def test_transfer() -> None:
    a, b = Account(100), Account(0)
    transfer(a, b, 30)
    assert (a.balance, b.balance) == (70, 30)


def test_failed_transfer_changes_nothing() -> None:
    a, b = Account(10), Account(0)
    with pytest.raises(InsufficientFunds):
        transfer(a, b, 50)
    assert (a.balance, b.balance) == (10, 0)
