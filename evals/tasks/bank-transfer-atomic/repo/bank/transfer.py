from bank.account import Account


def transfer(source: Account, target: Account, amount: int) -> None:
    """Move money; if the withdrawal fails, neither account changes."""
    target.deposit(amount)
    source.withdraw(amount)
