"""Session persistence (JSONL, one message per line) and project instructions."""

from __future__ import annotations

import datetime as dt
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletionMessageParam


def state_dir() -> Path:
    """Where sessions and traces live; ``AGENT_STATE_DIR`` overrides it."""
    override = os.environ.get("AGENT_STATE_DIR")
    return Path(override) if override else Path.home() / ".local" / "state" / "agent"


AGENTS_FILES = ("AGENTS.md", "CLAUDE.md")
MAX_PROJECT_INSTRUCTIONS = 6000


def project_instructions(workspace: Path) -> tuple[str, str] | None:
    """(file name, text) of the repo's agent instructions, if any.

    AGENTS.md is the open convention; CLAUDE.md is read as a fallback because
    many repositories already have one. Truncated to keep the prefix small.
    """
    for name in AGENTS_FILES:
        path = workspace / name
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if len(text) > MAX_PROJECT_INSTRUCTIONS:
                text = text[:MAX_PROJECT_INSTRUCTIONS] + "\n[... truncated ...]"
            return name, text
    return None


def build_system_prompt(base: str, workspace: Path) -> str:
    """Base prompt plus project instructions. Stable for a session, so the
    prefix stays cacheable: no timestamps, no per-turn content."""
    found = project_instructions(workspace)
    if found is None:
        return base
    name, text = found
    return f"{base}\n\nProject instructions from {name} (follow them):\n\n{text}"


@dataclass(frozen=True, slots=True)
class SessionInfo:
    id: str
    path: Path
    created: str
    workspace: str
    model: str
    messages: int


class SessionStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or state_dir() / "sessions"

    def new_id(self) -> str:
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{secrets.token_hex(2)}"

    def path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.jsonl"

    def save(
        self,
        session_id: str,
        messages: list[ChatCompletionMessageParam],
        meta: dict[str, Any],
    ) -> Path:
        """Rewrite the session file: a meta line, then one message per line."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path(session_id)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps({"meta": {"id": session_id, **meta}}) + "\n")
            for message in messages:
                fh.write(json.dumps(message, default=str) + "\n")
        tmp.replace(path)
        return path

    def load(self, session_id: str) -> tuple[dict[str, Any], list[ChatCompletionMessageParam]]:
        if session_id == "last":
            sessions = self.list()
            if not sessions:
                raise FileNotFoundError("no saved sessions")
            session_id = sessions[0].id
        path = self.path(session_id)
        lines = path.read_text(encoding="utf-8").splitlines()
        meta: dict[str, Any] = json.loads(lines[0])["meta"]
        messages: list[ChatCompletionMessageParam] = [
            json.loads(line) for line in lines[1:] if line
        ]
        return meta, messages

    def list(self, limit: int = 20) -> list[SessionInfo]:
        if not self.root.is_dir():
            return []
        infos = []
        for path in self.root.glob("*.jsonl"):
            with path.open(encoding="utf-8") as fh:
                first = fh.readline()
                count = sum(1 for _ in fh)
            meta = json.loads(first)["meta"]
            infos.append(
                SessionInfo(
                    id=meta["id"],
                    path=path,
                    created=meta.get("created", ""),
                    workspace=meta.get("workspace", ""),
                    model=meta.get("model", ""),
                    messages=count,
                )
            )
        return sorted(infos, key=lambda i: i.path.stat().st_mtime, reverse=True)[:limit]
