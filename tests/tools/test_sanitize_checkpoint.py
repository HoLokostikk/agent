from pathlib import Path

from agent.tools import EditFile, PathGuard, ToolContext, WriteFile
from agent.tools.checkpoint import CheckpointStack
from agent.tools.files import EditFileArgs, WriteFileArgs
from agent.tools.sanitize import sanitize


def test_sanitize_removes_escape_sequences_and_controls() -> None:
    text = "ok\x1b[31mred\x1b[0m\x1b]0;title\x07 bell\x07 tab\tnl\n\x00end\r\n"

    assert sanitize(text) == "okred bell tab\tnl\nend\n"


def test_sanitize_keeps_plain_text() -> None:
    assert sanitize("def f():\n    return 1\n") == "def f():\n    return 1\n"


async def test_checkpoint_restores_edits_and_removes_created_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    stack = CheckpointStack()
    ctx = ToolContext(PathGuard(tmp_path), None, stack)
    stack.begin("run 1")

    await EditFile().run(EditFileArgs(path="a.py", old_string="x = 1", new_string="x = 2"), ctx)
    await EditFile().run(EditFileArgs(path="a.py", old_string="x = 2", new_string="x = 3"), ctx)
    await WriteFile().run(WriteFileArgs(path="new.py", content="y = 1\n"), ctx)
    undone = stack.undo()

    assert undone is not None
    assert undone[0] == "run 1"
    assert (tmp_path / "a.py").read_text() == "x = 1\n"
    assert not (tmp_path / "new.py").exists()
    assert stack.undo() is None


async def test_undo_skips_runs_that_changed_nothing(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("v1\n")
    stack = CheckpointStack()
    ctx = ToolContext(PathGuard(tmp_path), None, stack)
    stack.begin("edit")
    await WriteFile().run(WriteFileArgs(path="a.py", content="v2\n"), ctx)
    stack.begin("question only")

    undone = stack.undo()

    assert undone is not None
    assert undone[0] == "edit"
    assert (tmp_path / "a.py").read_text() == "v1\n"
