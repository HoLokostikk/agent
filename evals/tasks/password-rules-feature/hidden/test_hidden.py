from passwords import validate_password


def test_all_rules_in_order() -> None:
    assert validate_password("a b") == ["length", "digit", "upper", "space"]


def test_tab_counts_as_space() -> None:
    assert validate_password("Secret123\t") == ["space"]
