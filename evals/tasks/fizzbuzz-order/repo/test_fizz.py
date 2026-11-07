from fizz import fizzbuzz


def test_plain() -> None:
    assert fizzbuzz(7) == "7"


def test_fizz_and_buzz() -> None:
    assert fizzbuzz(9) == "Fizz"
    assert fizzbuzz(10) == "Buzz"


def test_fizzbuzz() -> None:
    assert fizzbuzz(30) == "FizzBuzz"
