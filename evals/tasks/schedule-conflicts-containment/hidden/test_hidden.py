from schedule import Event, conflicts


def test_containment() -> None:
    events = [Event("inner", 70, 80), Event("outer", 60, 120)]
    assert conflicts(events) == {("inner", "outer")}


def test_three_way() -> None:
    events = [Event("a", 0, 100), Event("b", 10, 20), Event("c", 15, 30)]
    assert conflicts(events) == {("a", "b"), ("a", "c"), ("b", "c")}
