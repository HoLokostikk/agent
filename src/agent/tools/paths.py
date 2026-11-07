"""Path guard: the model may only touch files inside the workspace."""

from __future__ import annotations

from pathlib import Path, PurePosixPath


class PathError(Exception):
    pass


class PathGuard:
    """Resolves model-supplied paths and rejects anything outside ``root``.

    Paths are resolved with symlinks followed, so ``../`` tricks and symlinks
    the model created through ``bash`` (``ln -s /etc x``) both land outside
    the root and are refused. The container path ``/workspace/...`` is
    accepted as an alias of the root, since that is what ``bash`` shows.
    """

    def __init__(self, root: Path, container_root: str = "/workspace") -> None:
        self.root = root.resolve()
        self.container_root = PurePosixPath(container_root)

    def resolve(self, path: str) -> Path:
        if "\x00" in path:
            raise PathError("invalid path")
        # Models often send "" for "here"; treat it as the root instead of failing.
        path = path.strip() or "."
        candidate = PurePosixPath(path)
        if candidate.is_absolute():
            try:
                candidate = candidate.relative_to(self.container_root)
            except ValueError:
                raise PathError(
                    f"{path} is outside the workspace; use a path relative to the "
                    f"workspace root or under {self.container_root}"
                ) from None
        resolved = (self.root / candidate).resolve()
        if not resolved.is_relative_to(self.root):
            raise PathError(f"{path} resolves outside the workspace")
        return resolved

    def display(self, path: Path) -> str:
        """Workspace-relative form for messages to the model."""
        try:
            rel = path.resolve().relative_to(self.root)
        except ValueError:
            return str(path)
        return str(rel) if str(rel) != "." else "."
