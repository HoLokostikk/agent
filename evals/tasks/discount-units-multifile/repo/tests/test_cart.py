from shop.cart import checkout


def test_regular_customer() -> None:
    assert checkout([40.0, 60.0]) == 100.0


def test_member_gets_ten_percent_off() -> None:
    assert checkout([40.0, 60.0], member=True) == 90.0
