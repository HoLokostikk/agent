"""Security properties of the Docker sandbox.

``test_run_args_*`` run everywhere. The rest start real containers: run them
with ``make test-docker`` (and in the CI ``sandbox`` job).
"""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from agent.sandbox import DockerSandbox, build_image, image_exists


def test_run_args_lock_the_container_down(tmp_path: Path) -> None:
    args = DockerSandbox(tmp_path).run_args()
    joined = " ".join(args)

    assert "--network none" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--read-only" in args
    assert f"--user {os.getuid()}:{os.getgid()}" in joined
    assert f"{tmp_path.resolve()}:/workspace:rw" in joined
    assert "--pids-limit" in args
    assert "--memory" in args
    # Only the workspace is mounted: no docker socket, no home directory.
    assert sum(1 for a in args if a == "--volume") == 1


@pytest.fixture
async def sandbox(tmp_path: Path) -> AsyncIterator[DockerSandbox]:
    if not await image_exists():
        await build_image()
    (tmp_path / "hello.txt").write_text("hi\n")
    async with DockerSandbox(tmp_path) as box:
        yield box


@pytest.mark.docker
async def test_sees_the_workspace(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("pwd && cat hello.txt", 30)

    assert result.exit_code == 0
    assert result.output.splitlines() == ["/workspace", "hi"]


@pytest.mark.docker
async def test_writes_land_in_the_workspace_owned_by_the_user(sandbox: DockerSandbox) -> None:
    await sandbox.exec("echo out > made.txt", 30)

    made = sandbox.root / "made.txt"
    assert made.read_text() == "out\n"
    assert made.stat().st_uid == os.getuid()


@pytest.mark.docker
async def test_has_no_network(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec(
        "python -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\"", 30
    )

    assert result.exit_code != 0
    assert "Network is unreachable" in result.output or "OSError" in result.output


@pytest.mark.docker
async def test_is_not_root_and_cannot_write_outside_workspace(sandbox: DockerSandbox) -> None:
    whoami = await sandbox.exec("id -u", 30)
    etc = await sandbox.exec("touch /etc/owned", 30)
    usr = await sandbox.exec("touch /usr/owned", 30)

    assert whoami.output.strip() == str(os.getuid())
    assert whoami.output.strip() != "0"
    assert etc.exit_code != 0
    assert usr.exit_code != 0


@pytest.mark.docker
async def test_tmp_is_writable(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("echo x > /tmp/x && cat /tmp/x", 30)

    assert result.output.strip() == "x"


@pytest.mark.docker
async def test_command_timeout_kills_the_command(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("sleep 30", 1)

    assert result.timed_out
    assert result.exit_code == 124


@pytest.mark.docker
async def test_pytest_is_available(sandbox: DockerSandbox) -> None:
    result = await sandbox.exec("python -m pytest --version", 60)

    assert result.exit_code == 0
