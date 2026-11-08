def validate_password(password: str) -> list[str]:
    """Return the list of broken rules, empty if the password is valid.

    Rules, reported with exactly these names and in this order:
    "length"  - at least 8 characters
    "digit"   - at least one digit
    "upper"   - at least one uppercase letter
    "space"   - no whitespace anywhere
    """
    broken = []
    if len(password) < 8:
        broken.append("length")
    if not any(c.isdigit() for c in password):
        broken.append("digit")
    if not any(c.isupper() for c in password):
        broken.append("upper")
    if any(c.isspace() for c in password):
        broken.append("space")
    return broken
