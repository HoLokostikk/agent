from pathlib import Path

import pytest

from agent.tools import EditFile, Glob, Grep, ListDir, PathGuard, ReadFile, ToolContext, WriteFile
from agent.tools.files import (
    EditFileArgs,
    GlobArgs,
    GrepArgs,
    ListDirArgs,
    ReadFileArgs,
    WriteFileArgs,
)


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text(
        "def add(a, b):\n    return a - b\n\n\ndef sub(a, b):\n    return a - b\n"
    )
    (tmp_path / "README.md").write_text("hello\n")
    (tmp_path / ".env").write_text("API_KEY=sk-live-123\n")
    (tmp_path / ".env.example").write_text("API_KEY=\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("[core]\n")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    return ToolContext(PathGuard(tmp_path), sandbox=None)


async def test_read_file_numbers_lines(ctx: ToolContext) -> None:
    result = await ReadFile().run(ReadFileArgs(path="pkg/core.py"), ctx)

    assert result.ok
    assert "(lines 1-6 of 6)" in result.output
    assert "     2\t    return a - b" in result.output


async def test_read_file_pages(ctx: ToolContext) -> None:
    result = await ReadFile().run(ReadFileArgs(path="pkg/core.py", offset=2, limit=2), ctx)

    assert "(lines 2-3 of 6); continue with offset=4" in result.output
    assert "def add" not in result.output


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("missing.py", "does not exist"),
        ("pkg", "is a directory"),
        ("blob.bin", "binary"),
        ("../x", "outside"),
        (".env", "secrets"),
    ],
)
async def test_read_file_errors_are_results(ctx: ToolContext, path: str, message: str) -> None:
    result = await ReadFile().run(ReadFileArgs(path=path), ctx)

    assert not result.ok
    assert message in result.output


async def test_env_example_is_readable(ctx: ToolContext) -> None:
    result = await ReadFile().run(ReadFileArgs(path=".env.example"), ctx)

    assert result.ok


async def test_list_dir_hides_git_and_marks_dirs(ctx: ToolContext) -> None:
    result = await ListDir().run(ListDirArgs(path=""), ctx)

    assert result.ok
    lines = result.output.splitlines()
    assert lines[0] == "pkg/"
    assert ".git/" not in lines


async def test_glob_finds_files_and_rejects_escape(ctx: ToolContext) -> None:
    found = await Glob().run(GlobArgs(pattern="**/*.py"), ctx)
    escaped = await Glob().run(GlobArgs(pattern="../*"), ctx)

    assert found.output == "pkg/core.py"
    assert not escaped.ok


async def test_grep_reports_path_and_line(ctx: ToolContext) -> None:
    result = await Grep().run(GrepArgs(pattern=r"return a - b"), ctx)

    assert result.output.splitlines() == [
        "pkg/core.py:2: return a - b",
        "pkg/core.py:6: return a - b",
    ]


async def test_grep_never_reads_secret_files(ctx: ToolContext) -> None:
    result = await Grep().run(GrepArgs(pattern="API_KEY"), ctx)

    assert "sk-live-123" not in result.output
    assert ".env:" not in result.output
    assert ".env.example:1" in result.output  # templates stay searchable


async def test_grep_bad_regex_is_an_error(ctx: ToolContext) -> None:
    result = await Grep().run(GrepArgs(pattern="("), ctx)

    assert not result.ok
    assert "invalid regular expression" in result.output


async def test_edit_requires_unique_match(ctx: ToolContext) -> None:
    result = await EditFile().run(
        EditFileArgs(path="pkg/core.py", old_string="return a - b", new_string="return a + b"), ctx
    )

    assert not result.ok
    assert "occurs 2 times" in result.output


async def test_edit_with_context_applies_once(ctx: ToolContext) -> None:
    args = EditFileArgs(
        path="pkg/core.py",
        old_string="def add(a, b):\n    return a - b",
        new_string="def add(a, b):\n    return a + b",
    )

    preview = EditFile().preview(args, ctx)
    result = await EditFile().run(args, ctx)

    assert preview is not None
    assert "-    return a - b" in preview
    assert "+    return a + b" in preview
    assert result.ok
    text = (ctx.guard.root / "pkg" / "core.py").read_text()
    assert text.count("return a + b") == 1
    assert text.count("return a - b") == 1


async def test_edit_not_found_tells_model_what_to_do(ctx: ToolContext) -> None:
    result = await EditFile().run(
        EditFileArgs(path="pkg/core.py", old_string="return a * b", new_string="x"), ctx
    )

    assert not result.ok
    assert "read the file again" in result.output


async def test_edit_preserves_crlf(ctx: ToolContext) -> None:
    path = ctx.guard.root / "win.txt"
    path.write_bytes(b"one\r\ntwo\r\n")

    await EditFile().run(EditFileArgs(path="win.txt", old_string="two", new_string="2"), ctx)

    assert path.read_bytes() == b"one\r\n2\r\n"


async def test_write_creates_parents_and_previews_diff(ctx: ToolContext) -> None:
    args = WriteFileArgs(path="new/dir/x.py", content="print(1)\n")

    preview = WriteFile().preview(args, ctx)
    result = await WriteFile().run(args, ctx)

    assert preview is not None
    assert "+print(1)" in preview
    assert result.ok
    assert "created new/dir/x.py" in result.output
    assert (ctx.guard.root / "new" / "dir" / "x.py").read_text() == "print(1)\n"
