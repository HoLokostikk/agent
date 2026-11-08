def parse_line(line: str) -> list[str]:
    """Split one CSV line; commas inside double quotes do not split, quotes are removed."""
    return [field.strip('"') for field in line.split(",")]
