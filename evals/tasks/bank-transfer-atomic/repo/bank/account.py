class InsufficientFunds(Exception):
    pass


class Account:
    def __init__(self, balance: int = 0) -> None:
        self.balance = balance

    def deposit(self, amount: int) -> None:
        self.balance += amount

    def withdraw(self, amount: int) -> None:
        """Raise InsufficientFunds when the balance is too low; never go negative."""
        self.balance -= amount
