from pathlib import Path

import pytest

from agent.tools import PathError, PathGuard


@pytest.fixture
def guard(tmp_path: Path) -> PathGuard:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    return PathGuard(tmp_path)


def test_relative_path_resolves_inside_root(guard: PathGuard) -> None:
    assert guard.resolve("src/a.py") == guard.root / "src" / "a.py"


def test_container_path_is_an_alias_of_root(guard: PathGuard) -> None:
    assert guard.resolve("/workspace/src/a.py") == guard.root / "src" / "a.py"


@pytest.mark.parametrize("path", ["", "  ", ".", "/workspace"])
def test_empty_and_dot_mean_root(guard: PathGuard, path: str) -> None:
    assert guard.resolve(path) == guard.root


@pytest.mark.parametrize(
    "path", ["../outside", "src/../../outside", "/etc/passwd", "/workspace/../etc"]
)
def test_escapes_are_refused(guard: PathGuard, path: str) -> None:
    with pytest.raises(PathError):
        guard.resolve(path)


def test_symlink_out_of_root_is_refused(
    guard: PathGuard, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "secret.txt").write_text("secret")
    (guard.root / "link").symlink_to(outside)

    with pytest.raises(PathError, match="outside"):
        guard.resolve("link/secret.txt")


def test_null_byte_is_refused(guard: PathGuard) -> None:
    with pytest.raises(PathError):
        guard.resolve("a\x00b")
