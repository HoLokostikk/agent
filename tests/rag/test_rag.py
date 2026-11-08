import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from agent.evals.retrieval import case_metrics, cases_from_git, evaluate, ranked_files
from agent.rag.bm25 import BM25, tokenize
from agent.rag.chunker import chunk_file
from agent.rag.embed import Vectors, normalize
from agent.rag.index import CodeIndex
from agent.rag.search import Method, Searcher, rrf
from agent.rag.tool import SearchCode, SearchCodeArgs
from agent.tools import PathGuard, ToolContext

SOURCE = '''"""Config loading."""
import json

DEFAULT = {"debug": False}


def parse_config(text: str) -> dict:
    return json.loads(text)


class Loader:
    def load(self, path: str) -> dict:
        return parse_config(open(path).read())
'''


class HashEmbedder:
    """Bag-of-tokens vectors: deterministic, and similar texts get similar vectors."""

    name = "hash@1"

    async def embed(self, texts: Sequence[str]) -> Vectors:
        rows = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                rows[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % 64] += 1  # noqa: S324
        return normalize(rows)


def test_python_chunks_follow_definitions() -> None:
    chunks = chunk_file("cfg.py", SOURCE)

    symbols = {c.symbol: (c.start_line, c.end_line) for c in chunks}
    assert symbols["func parse_config"] == (7, 8)
    assert symbols["class Loader"] == (11, 13)
    assert "module" in symbols


def test_large_classes_are_split_by_method() -> None:
    methods = "\n".join(
        f"    def m{i}(self) -> int:\n" + "        x = 1\n" * 10 + "        return x\n"
        for i in range(8)
    )
    chunks = chunk_file("big.py", f"class Big:\n{methods}")

    assert {c.symbol for c in chunks} >= {"Big.m0", "Big.m7"}


def test_broken_python_falls_back_to_windows() -> None:
    chunks = chunk_file("bad.py", "def (:\n" * 100)

    assert chunks[0].symbol.startswith("lines 1-")


def test_tokenize_splits_identifiers() -> None:
    assert set(tokenize("parseConfig")) >= {"parse", "config", "parseconfig"}
    assert set(tokenize("parse_config")) >= {"parse", "config"}
    assert "the" not in tokenize("the HTTPServer")
    assert {"http", "server"} <= set(tokenize("HTTPServer"))


def test_bm25_prefers_matching_document() -> None:
    bm25 = BM25(["def parse_config(): ...", "def render_html(): ...", "class Cart: ..."])

    scores = bm25.scores("how is the config parsed")

    assert scores.index(max(scores)) == 0


def test_rrf_rewards_agreement() -> None:
    assert rrf([[1, 2, 3], [3, 1, 2]])[0] == 1
    assert rrf([[5], [6]]) == [5, 6]


def write_repo(root: Path) -> None:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "config.py").write_text(SOURCE)
    (root / "pkg" / "cart.py").write_text("def checkout(prices):\n    return sum(prices)\n")
    (root / ".env").write_text("API_KEY=sk-live\n")
    (root / "notes.bin").write_bytes(b"\x00\x01")


async def test_index_is_incremental_and_skips_secrets(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo(repo)
    index = CodeIndex(repo, tmp_path / "idx.sqlite")

    first = await index.update(HashEmbedder())
    unchanged = await index.update(HashEmbedder())
    (repo / "pkg" / "cart.py").write_text(
        "def checkout(prices, member=False):\n    return sum(prices)\n"
    )
    (repo / "pkg" / "config.py").unlink()
    after = await index.update(HashEmbedder())

    assert first.files_changed == 2
    assert unchanged.files_changed == 0
    assert (after.files_changed, after.files_deleted) == (1, 1)
    paths = {c.path for c in index.chunks()}
    assert paths == {"pkg/cart.py"}
    assert all("sk-live" not in c.text for c in index.chunks())
    index.close()


async def test_changing_the_embedder_reembeds_everything(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo(repo)
    index = CodeIndex(repo, tmp_path / "idx.sqlite")
    await index.update(None)
    _, vectors = index.vectors()
    assert vectors is None

    stats = await index.update(HashEmbedder())
    _, vectors = index.vectors()

    assert stats.files_changed == 2
    assert vectors is not None
    assert index.embedder_name == "hash@1"


def test_index_respects_gitignore(tmp_path: Path) -> None:
    import asyncio

    repo = tmp_path / "repo"
    write_repo(repo)
    (repo / "generated.py").write_text("x = 1\n")
    (repo / ".gitignore").write_text("generated.py\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    index = CodeIndex(repo, tmp_path / "idx.sqlite")

    asyncio.run(index.update(None))

    assert "generated.py" not in {c.path for c in index.chunks()}


async def test_hybrid_search_and_tool(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    write_repo(repo)
    index = CodeIndex(repo, tmp_path / "idx.sqlite")
    await index.update(HashEmbedder())
    chunks, vectors = index.vectors()
    searcher = Searcher(chunks, vectors, HashEmbedder())

    assert searcher.methods == [Method.BM25, Method.VECTOR, Method.HYBRID]
    for method in searcher.methods:
        hits = await searcher.search("parse config", 3, method)
        assert hits[0].chunk.path == "pkg/config.py", method

    result = await SearchCode(searcher).run(
        SearchCodeArgs(query="checkout total"), ToolContext(PathGuard(repo), None)
    )
    assert result.output.splitlines()[0].startswith("pkg/cart.py:1-2")


async def test_unavailable_method_is_an_error() -> None:
    searcher = Searcher(chunk_file("a.py", SOURCE))

    with pytest.raises(ValueError, match="not available"):
        await searcher.search("x", method=Method.VECTOR)


def test_case_metrics() -> None:
    files = ranked_files(["a.py", "a.py", "b.py", "c.py"])

    assert files == ["a.py", "b.py", "c.py"]
    m = case_metrics(files, frozenset({"b.py", "z.py"}))
    assert m == {"recall@5": 0.5, "recall@10": 0.5, "mrr": 0.5}


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


async def test_retrieval_eval_on_git_history(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    (repo / "config.py").write_text(SOURCE)
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Add config parsing with JSON")
    (repo / "cart.py").write_text("def checkout(prices):\n    return sum(prices)\n")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "Add cart checkout total")
    git(repo, "commit", "-q", "--allow-empty", "-m", "bump version")

    cases = cases_from_git(repo)
    index = CodeIndex(repo, tmp_path / "idx.sqlite")
    await index.update(None)
    result = await evaluate(Searcher(index.chunks()), cases)

    assert [c.query for c in cases] == ["Add cart checkout total", "Add config parsing with JSON"]
    bm25 = result["summary"]["methods"]["bm25"]
    assert bm25["recall@5"] == 1.0
    assert bm25["mrr"] == 1.0
