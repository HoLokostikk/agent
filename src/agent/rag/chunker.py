"""Split source files into retrievable chunks with line ranges."""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path

WINDOW_LINES = 60
WINDOW_OVERLAP = 10
MAX_CHUNK_LINES = 200


@dataclass(frozen=True, slots=True)
class Chunk:
    path: str  # workspace-relative, posix
    start_line: int  # 1-based, inclusive
    end_line: int
    symbol: str  # "module", "func parse_config", "class Cart", "Cart.checkout", "lines 1-60"
    text: str

    @property
    def id(self) -> str:
        raw = f"{self.path}:{self.start_line}:{self.end_line}:{self.symbol}"
        return hashlib.sha1(raw.encode(), usedforsecurity=False).hexdigest()[:16]

    def header(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line} {self.symbol}"


def _windows(path: str, lines: list[str]) -> list[Chunk]:
    chunks = []
    step = WINDOW_LINES - WINDOW_OVERLAP
    for start in range(0, max(1, len(lines)), step):
        end = min(len(lines), start + WINDOW_LINES)
        text = "\n".join(lines[start:end]).strip()
        if text:
            chunks.append(Chunk(path, start + 1, end, f"lines {start + 1}-{end}", text))
        if end >= len(lines):
            break
    return chunks


def _python(path: str, source: str) -> list[Chunk]:
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _windows(path, lines)
    chunks: list[Chunk] = []
    covered: set[int] = set()

    def add(node: ast.AST, symbol: str) -> None:
        start = node.lineno  # type: ignore[attr-defined]
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, *(d.lineno for d in decorators))
        end = node.end_lineno or start  # type: ignore[attr-defined]
        if end - start + 1 > MAX_CHUNK_LINES:
            for window in _windows(path, lines[start - 1 : end]):
                chunks.append(
                    Chunk(
                        path,
                        window.start_line + start - 1,
                        window.end_line + start - 1,
                        f"{symbol} (part)",
                        window.text,
                    )
                )
        else:
            chunks.append(Chunk(path, start, end, symbol, "\n".join(lines[start - 1 : end])))
        covered.update(range(start, end + 1))

    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            add(node, f"func {node.name}")
        elif isinstance(node, ast.ClassDef):
            methods = [
                n for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
            ]
            if (
                len(lines[node.lineno - 1 : node.end_lineno or node.lineno]) <= WINDOW_LINES
                or not methods
            ):
                add(node, f"class {node.name}")
            else:
                for method in methods:
                    add(method, f"{node.name}.{method.name}")
    # Module-level code outside functions and classes: imports, constants, scripts.
    rest = [i for i in range(1, len(lines) + 1) if i not in covered and lines[i - 1].strip()]
    if rest:
        text = "\n".join(lines[i - 1] for i in rest[:WINDOW_LINES])
        chunks.append(Chunk(path, rest[0], rest[min(len(rest), WINDOW_LINES) - 1], "module", text))
    return chunks


def chunk_file(path: str, source: str) -> list[Chunk]:
    if path.endswith(".py"):
        return _python(path, source)
    return _windows(path, source.splitlines())


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


TEXT_SUFFIXES = frozenset(
    {
        ".py",
        ".md",
        ".rst",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".json",
        ".cfg",
        ".ini",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".go",
        ".rs",
        ".java",
        ".kt",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".sh",
        ".sql",
        ".html",
        ".css",
        ".swift",
        ".scala",
    }
)


def is_indexable(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES
