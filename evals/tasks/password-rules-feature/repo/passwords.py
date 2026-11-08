def validate_password(password: str) -> list[str]:
    """Return the list of broken rules, empty if the password is valid.

    Rules, reported with exactly these names and in this order:
    "length"  - at least 8 characters
    "digit"   - at least one digit
    "upper"   - at least one uppercase letter
    "space"   - no whitespace anywhere
    """
    raise NotImplementedError
