def parse_query(qs: str) -> dict[str, str]:
    """Parse 'a=1&b=two+words' into a dict; values are URL-decoded ('+' is a space)."""
    result: dict[str, str] = {}
    for pair in qs.split("&"):
        if not pair:
            continue
        key, _, value = pair.partition("=")
        result[key] = value
    return result
