"""File checkpoints for /undo: the original bytes of every file a run changes.

Only edits made through edit_file / write_file are tracked. Changes made by
bash commands inside the sandbox are not; the REPL says so when undoing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Checkpoint:
    label: str
    originals: dict[Path, bytes | None] = field(default_factory=dict)  # None: did not exist

    def remember(self, path: Path) -> None:
        if path not in self.originals:
            self.originals[path] = path.read_bytes() if path.is_file() else None

    def restore(self) -> list[Path]:
        restored = []
        for path, data in self.originals.items():
            if data is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            restored.append(path)
        return restored


@dataclass
class CheckpointStack:
    checkpoints: list[Checkpoint] = field(default_factory=list)

    def begin(self, label: str) -> None:
        self.checkpoints.append(Checkpoint(label))

    def remember(self, path: Path) -> None:
        if not self.checkpoints:
            self.begin("implicit")
        self.checkpoints[-1].remember(path)

    def undo(self) -> tuple[str, list[Path]] | None:
        while self.checkpoints:
            checkpoint = self.checkpoints.pop()
            if checkpoint.originals:
                return checkpoint.label, checkpoint.restore()
        return None
