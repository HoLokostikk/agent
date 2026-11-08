"""File tools. They run on the host, confined to the workspace by PathGuard."""

from __future__ import annotations

import difflib
import fnmatch
import re
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from pydantic import Field

from agent.tools.base import Tool, ToolArgs, ToolContext, ToolKind, ToolResult
from agent.tools.paths import PathError, PathGuard
from agent.tools.truncate import head_tail

IGNORED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        "dist",
        "build",
    }
)
SECRET_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    "*.p12",
    "*.pfx",
    ".netrc",
    ".pypirc",
    "credentials*",
)
SECRET_EXCEPTIONS = (".env.example", ".env.sample", ".env.template")
MAX_FILE_BYTES = 2_000_000
MAX_LINE_CHARS = 400


def is_secret_file(path: Path) -> bool:
    name = path.name
    if name in SECRET_EXCEPTIONS:
        return False
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in SECRET_PATTERNS)


def _is_binary(path: Path) -> bool:
    with path.open("rb") as fh:
        return b"\x00" in fh.read(8192)


def _read_text(path: Path) -> str:
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        return fh.read()


def _write_text(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def _clip(line: str) -> str:
    return line if len(line) <= MAX_LINE_CHARS else line[:MAX_LINE_CHARS] + " [...]"


def _numbered(lines: list[str], start: int) -> str:
    return "\n".join(f"{n:>6}\t{_clip(line)}" for n, line in enumerate(lines, start))


def _readable_file(guard: PathGuard, raw: str) -> Path:
    path = guard.resolve(raw)
    if not path.exists():
        raise PathError(f"{raw} does not exist")
    if path.is_dir():
        raise PathError(f"{raw} is a directory; use list_dir")
    if is_secret_file(path):
        raise PathError(f"{raw} may contain secrets and cannot be read")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise PathError(f"{raw} is larger than {MAX_FILE_BYTES} bytes")
    if _is_binary(path):
        raise PathError(f"{raw} looks like a binary file")
    return path


def _walk_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if any(part in IGNORED_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_file():
            yield path


def _diff(path_label: str, before: str, after: str) -> str:
    diff = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{path_label}",
        tofile=f"b/{path_label}",
    )
    return "".join(diff) or "(no changes)"


class ReadFileArgs(ToolArgs):
    path: str = Field(description="File path relative to the workspace root.")
    offset: int = Field(default=1, ge=1, description="First line to show, 1-based.")
    limit: int = Field(default=200, ge=1, le=1000, description="Maximum number of lines.")


class ReadFile(Tool):
    name: ClassVar[str] = "read_file"
    description: ClassVar[str] = (
        "Read a text file. Returns lines prefixed with their line numbers. "
        "Use offset and limit for long files."
    )
    kind: ClassVar[ToolKind] = ToolKind.READ
    Args: ClassVar[type[ToolArgs]] = ReadFileArgs

    async def run(self, args: ReadFileArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = _readable_file(ctx.guard, args.path)
        except PathError as exc:
            return ToolResult(False, str(exc))
        lines = _read_text(path).splitlines()
        total = len(lines)
        if total == 0:
            return ToolResult(True, f"{args.path} is empty")
        if args.offset > total:
            return ToolResult(False, f"offset {args.offset} is past the end ({total} lines)")
        end = min(total, args.offset + args.limit - 1)
        body = _numbered(lines[args.offset - 1 : end], args.offset)
        header = f"{ctx.guard.display(path)} (lines {args.offset}-{end} of {total})"
        if end < total:
            header += f"; continue with offset={end + 1}"
        return ToolResult(True, f"{header}\n{body}")


class ListDirArgs(ToolArgs):
    path: str = Field(default=".", description="Directory relative to the workspace root.")


class ListDir(Tool):
    name: ClassVar[str] = "list_dir"
    description: ClassVar[str] = "List a directory. Directories end with '/'."
    kind: ClassVar[ToolKind] = ToolKind.READ
    Args: ClassVar[type[ToolArgs]] = ListDirArgs

    async def run(self, args: ListDirArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.guard.resolve(args.path)
        except PathError as exc:
            return ToolResult(False, str(exc))
        if not path.is_dir():
            return ToolResult(False, f"{args.path} is not a directory")
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
            if child.name in IGNORED_DIRS:
                continue
            entries.append(child.name + ("/" if child.is_dir() else ""))
        if not entries:
            return ToolResult(True, f"{args.path} is empty")
        return ToolResult(True, head_tail("\n".join(entries), max_lines=200))


class GlobArgs(ToolArgs):
    pattern: str = Field(description="Glob relative to the workspace root, e.g. '**/*.py'.")


class Glob(Tool):
    name: ClassVar[str] = "glob"
    description: ClassVar[str] = "Find files by glob pattern, e.g. '**/test_*.py'."
    kind: ClassVar[ToolKind] = ToolKind.READ
    Args: ClassVar[type[ToolArgs]] = GlobArgs
    max_results: ClassVar[int] = 200

    async def run(self, args: GlobArgs, ctx: ToolContext) -> ToolResult:
        if args.pattern.startswith("/") or ".." in args.pattern.split("/"):
            return ToolResult(False, "pattern must be relative and must not contain '..'")
        root = ctx.guard.root
        matches = []
        for path in sorted(root.glob(args.pattern)):
            rel = path.relative_to(root)
            if any(part in IGNORED_DIRS for part in rel.parts):
                continue
            if not path.resolve().is_relative_to(root) or not path.is_file():
                continue
            matches.append(str(rel))
        if not matches:
            return ToolResult(True, f"no files match {args.pattern}")
        shown = matches[: self.max_results]
        extra = len(matches) - len(shown)
        tail = f"\n[... {extra} more]" if extra else ""
        return ToolResult(True, "\n".join(shown) + tail)


class GrepArgs(ToolArgs):
    pattern: str = Field(description="Python regular expression.")
    path: str = Field(default=".", description="File or directory to search.")
    include: str | None = Field(default=None, description="Only files matching, e.g. '*.py'.")
    ignore_case: bool = False


class Grep(Tool):
    name: ClassVar[str] = "grep"
    description: ClassVar[str] = (
        "Search file contents with a regular expression. Returns path:line: text."
    )
    kind: ClassVar[ToolKind] = ToolKind.READ
    Args: ClassVar[type[ToolArgs]] = GrepArgs
    max_matches: ClassVar[int] = 100

    async def run(self, args: GrepArgs, ctx: ToolContext) -> ToolResult:
        try:
            regex = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
        except re.error as exc:
            return ToolResult(False, f"invalid regular expression: {exc}")
        try:
            target = ctx.guard.resolve(args.path)
        except PathError as exc:
            return ToolResult(False, str(exc))
        if not target.exists():
            return ToolResult(False, f"{args.path} does not exist")
        files = [target] if target.is_file() else list(_walk_files(target))
        matches: list[str] = []
        total = 0
        for path in files:
            if args.include and not fnmatch.fnmatch(path.name, args.include):
                continue
            if is_secret_file(path) or path.stat().st_size > MAX_FILE_BYTES or _is_binary(path):
                continue
            label = ctx.guard.display(path)
            for n, line in enumerate(_read_text(path).splitlines(), 1):
                if regex.search(line):
                    total += 1
                    if len(matches) < self.max_matches:
                        matches.append(f"{label}:{n}: {_clip(line.strip())}")
        if not matches:
            return ToolResult(True, f"no matches for {args.pattern!r}")
        extra = f"\n[... {total - len(matches)} more matches]" if total > len(matches) else ""
        return ToolResult(True, "\n".join(matches) + extra)


class EditFileArgs(ToolArgs):
    path: str = Field(description="File to edit, relative to the workspace root.")
    old_string: str = Field(
        description="Exact text to replace, including whitespace. Must occur exactly once."
    )
    new_string: str = Field(description="Replacement text.")


class EditFile(Tool):
    name: ClassVar[str] = "edit_file"
    description: ClassVar[str] = (
        "Replace one exact occurrence of old_string with new_string in a file. "
        "old_string must match the file exactly and be unique; include surrounding "
        "lines if needed. Read the file first."
    )
    kind: ClassVar[ToolKind] = ToolKind.WRITE
    Args: ClassVar[type[ToolArgs]] = EditFileArgs

    def _plan(self, args: EditFileArgs, guard: PathGuard) -> tuple[Path, str, str]:
        path = _readable_file(guard, args.path)
        if args.old_string == args.new_string:
            raise PathError("old_string and new_string are identical; nothing to change")
        if not args.old_string:
            raise PathError("old_string is empty; use write_file to create a file")
        before = _read_text(path)
        count = before.count(args.old_string)
        if count == 0:
            raise PathError(
                f"old_string not found in {args.path}; read the file again and copy the "
                "text exactly, including indentation"
            )
        if count > 1:
            raise PathError(
                f"old_string occurs {count} times in {args.path}; include more surrounding "
                "lines so it matches exactly once"
            )
        return path, before, before.replace(args.old_string, args.new_string, 1)

    def preview(self, args: EditFileArgs, ctx: ToolContext) -> str | None:
        try:
            path, before, after = self._plan(args, ctx.guard)
        except PathError as exc:
            return f"(edit will fail: {exc})"
        return _diff(ctx.guard.display(path), before, after)

    async def run(self, args: EditFileArgs, ctx: ToolContext) -> ToolResult:
        try:
            path, before, after = self._plan(args, ctx.guard)
        except PathError as exc:
            return ToolResult(False, str(exc))
        if ctx.checkpoints is not None:
            ctx.checkpoints.remember(path)
        _write_text(path, after)
        start = before[: before.index(args.old_string)].count("\n") + 1
        new_lines = after.splitlines()
        changed = max(1, args.new_string.count("\n") + 1)
        lo = max(1, start - 2)
        hi = min(len(new_lines), start + changed + 1)
        snippet = _numbered(new_lines[lo - 1 : hi], lo)
        return ToolResult(True, f"edited {ctx.guard.display(path)}:\n{snippet}")


class WriteFileArgs(ToolArgs):
    path: str = Field(description="File to create or overwrite, relative to the workspace root.")
    content: str = Field(description="Full new content of the file.")


class WriteFile(Tool):
    name: ClassVar[str] = "write_file"
    description: ClassVar[str] = (
        "Create a file or overwrite it completely. Prefer edit_file for changes to an "
        "existing file."
    )
    kind: ClassVar[ToolKind] = ToolKind.WRITE
    Args: ClassVar[type[ToolArgs]] = WriteFileArgs

    def preview(self, args: WriteFileArgs, ctx: ToolContext) -> str | None:
        try:
            path = ctx.guard.resolve(args.path)
        except PathError as exc:
            return f"(write will fail: {exc})"
        before = _read_text(path) if path.is_file() else ""
        return _diff(ctx.guard.display(path), before, args.content)

    async def run(self, args: WriteFileArgs, ctx: ToolContext) -> ToolResult:
        try:
            path = ctx.guard.resolve(args.path)
        except PathError as exc:
            return ToolResult(False, str(exc))
        if path.is_dir():
            return ToolResult(False, f"{args.path} is a directory")
        existed = path.exists()
        if ctx.checkpoints is not None:
            ctx.checkpoints.remember(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_text(path, args.content)
        lines = args.content.count("\n") + (0 if args.content.endswith("\n") else 1)
        verb = "overwrote" if existed else "created"
        return ToolResult(True, f"{verb} {ctx.guard.display(path)} ({lines} lines)")
