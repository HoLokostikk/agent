"""Incremental code index: SQLite for chunks and file hashes, vectors as blobs.

Freshness: files are re-chunked only when their content hash changes, and the
chunks of deleted files are removed. Access control: secret files, ignored
directories and anything gitignored never enter the index.
"""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agent.core.session import state_dir
from agent.rag.chunker import Chunk, chunk_file, file_hash, is_indexable
from agent.rag.embed import Embedder, Vectors
from agent.tools.files import IGNORED_DIRS, MAX_FILE_BYTES, is_secret_file

SCHEMA = """
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY, path TEXT NOT NULL, start_line INT, end_line INT,
    symbol TEXT, text TEXT, vector BLOB
);
CREATE INDEX IF NOT EXISTS chunks_path ON chunks(path);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass(frozen=True, slots=True)
class UpdateStats:
    files_seen: int
    files_changed: int
    files_deleted: int
    chunks_added: int
    chunks_total: int


def default_index_path(workspace: Path) -> Path:
    digest = hashlib.sha1(str(workspace.resolve()).encode(), usedforsecurity=False).hexdigest()[:12]
    return state_dir() / "index" / f"{workspace.resolve().name}-{digest}.sqlite"


def candidate_files(root: Path) -> Iterator[Path]:
    """Files worth indexing: git-tracked when in a repo (respects .gitignore)."""
    tracked: list[Path] | None = None
    if (root / ".git").exists():
        try:
            out = subprocess.run(
                ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                cwd=root,
                capture_output=True,
                check=True,
            ).stdout
            tracked = [root / p for p in out.decode().split("\0") if p]
        except (OSError, subprocess.CalledProcessError):
            tracked = None
    paths = tracked if tracked is not None else sorted(root.rglob("*"))
    for path in paths:
        rel = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if (
            not path.is_file()
            or path.is_symlink()
            or is_secret_file(path)
            or not is_indexable(path)
        ):
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        yield path


class CodeIndex:
    def __init__(self, root: Path, path: Path | None = None) -> None:
        self.root = root.resolve()
        self.path = path or default_index_path(self.root)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)

    def close(self) -> None:
        self.db.close()

    def _meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    async def update(self, embedder: Embedder | None = None) -> UpdateStats:
        if embedder is not None and self._meta("embedder") != embedder.name:
            self.db.execute("DELETE FROM files")
            self.db.execute("DELETE FROM chunks")
        known = dict(self.db.execute("SELECT path, hash FROM files").fetchall())
        seen: set[str] = set()
        changed: list[tuple[str, str, list[Chunk]]] = []
        for path in candidate_files(self.root):
            rel = path.relative_to(self.root).as_posix()
            seen.add(rel)
            data = path.read_bytes()
            digest = file_hash(data)
            if known.get(rel) == digest:
                continue
            text = data.decode("utf-8", errors="replace")
            changed.append((rel, digest, chunk_file(rel, text)))
        deleted = set(known) - seen
        new_chunks = [c for _, _, chunks in changed for c in chunks]
        vectors: Vectors | None = None
        if embedder is not None and new_chunks:
            vectors = await embedder.embed([f"{c.header()}\n{c.text}" for c in new_chunks])
        with self.db:
            for rel in deleted | {rel for rel, _, _ in changed}:
                self.db.execute("DELETE FROM chunks WHERE path = ?", (rel,))
            for rel in deleted:
                self.db.execute("DELETE FROM files WHERE path = ?", (rel,))
            for i, c in enumerate(new_chunks):
                blob = vectors[i].tobytes() if vectors is not None else None
                self.db.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (c.id, c.path, c.start_line, c.end_line, c.symbol, c.text, blob),
                )
            for rel, digest, _ in changed:
                self.db.execute("INSERT OR REPLACE INTO files VALUES (?, ?)", (rel, digest))
            if embedder is not None:
                self.db.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (embedder.name,)
                )
        total = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return UpdateStats(len(seen), len(changed), len(deleted), len(new_chunks), total)

    def chunks(self) -> list[Chunk]:
        rows = self.db.execute(
            "SELECT path, start_line, end_line, symbol, text FROM chunks ORDER BY path, start_line"
        ).fetchall()
        return [Chunk(*row) for row in rows]

    def vectors(self) -> tuple[list[Chunk], Vectors | None]:
        rows = self.db.execute(
            "SELECT path, start_line, end_line, symbol, text, vector FROM chunks "
            "ORDER BY path, start_line"
        ).fetchall()
        chunks = [Chunk(*row[:5]) for row in rows]
        blobs = [row[5] for row in rows]
        if not blobs or any(b is None for b in blobs):
            return chunks, None
        return chunks, np.stack([np.frombuffer(b, dtype=np.float32) for b in blobs])

    @property
    def embedder_name(self) -> str | None:
        return self._meta("embedder")
