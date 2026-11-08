def parse_line(line: str) -> list[str]:
    """Split one CSV line; commas inside double quotes do not split, quotes are removed."""
    import csv

    return next(csv.reader([line]))
