from schedule import Event, conflicts


def test_simple_overlap() -> None:
    events = [Event("a", 60, 120), Event("b", 90, 150)]
    assert conflicts(events) == {("a", "b")}


def test_back_to_back_is_fine() -> None:
    assert conflicts([Event("a", 60, 120), Event("b", 120, 180)]) == set()


def test_later_event_listed_first() -> None:
    events = [Event("late", 90, 150), Event("early", 60, 120)]
    assert conflicts(events) == {("early", "late")}
