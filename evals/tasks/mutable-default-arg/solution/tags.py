def add_tag(tag: str, tags: list[str] | None = None) -> list[str]:
    """Return a list with `tag` appended; a new list when `tags` is omitted."""
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
