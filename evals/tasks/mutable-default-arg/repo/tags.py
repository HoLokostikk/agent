def add_tag(tag: str, tags: list[str] = []) -> list[str]:  # noqa: B006
    """Return a list with `tag` appended; a new list when `tags` is omitted."""
    tags.append(tag)
    return tags
