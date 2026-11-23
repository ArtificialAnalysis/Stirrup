"""Live tests for BoxdCodeExecToolProvider.

Skipped unless ``BOXD_E2E=1``. They need ``BOXD_API_KEY``, create real machines
and delete them on exit. ``BOXD_E2E_PREFIX`` sets the machine name prefix and
``BOXD_E2E_REGION`` the region.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import anyio
import pytest

pytest.importorskip("boxd")

from boxd import AsyncBoxd
from boxd.errors import AuthenticationError, NotFoundError

from stirrup.tools.code_backends.boxd import BOXD_WORKING_DIR, BoxdCodeExecToolProvider

pytestmark = [
    pytest.mark.boxd,
    pytest.mark.skipif(os.environ.get("BOXD_E2E") != "1", reason="set BOXD_E2E=1 to run live boxd tests"),
]


def make_provider(**kwargs: object) -> BoxdCodeExecToolProvider:
    return BoxdCodeExecToolProvider(
        name_prefix=os.environ.get("BOXD_E2E_PREFIX", "stirrup-e2e-"),
        region=os.environ.get("BOXD_E2E_REGION", "nyc"),
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


async def test_commands_files_and_isolation(tmp_path: Path) -> None:
    provider = make_provider(env={"STIRRUP_E2E": "reached"})
    async with provider:
        result = await provider.run_command("echo out; echo err >&2; exit 7")
        assert (result.exit_code, result.stdout, result.stderr) == (7, "out\n", "err\n")

        assert (await provider.run_command("[[ a == a ]] && pwd")).stdout.strip() == BOXD_WORKING_DIR
        assert (await provider.run_command('echo "$STIRRUP_E2E"')).stdout.strip() == "reached"
        assert (await provider.run_command("cat; echo closed", timeout=10)).stdout.strip() == "closed"

        timed_out = await provider.run_command("sleep 60", timeout=2)
        assert timed_out.error_kind == "timeout"

        absent = await provider.run_command("command -v boxd run; echo found=$?")
        assert absent.stdout.strip() == "found=1"
        internet = await provider.run_command("curl -s -m 10 -o /dev/null -w '%{http_code}' https://pypi.org/simple/")
        assert internet.stdout == "200"

        await provider.run_command("nohup sleep 30 >/dev/null 2>&1 &")
        assert (await provider.run_command("pgrep -x sleep >/dev/null; echo $?")).stdout.strip() == "0"

        payload = os.urandom(5 * 1024 * 1024)
        await provider.write_file_bytes("nested/big.bin", payload)
        back = await provider.read_file_bytes("nested/big.bin")
        assert hashlib.sha256(back).digest() == hashlib.sha256(payload).digest()
        assert await provider.is_directory("nested")
        assert await provider.file_exists("nested/big.bin")
        assert await provider.list_files("nested") == ["big.bin"]
        with pytest.raises(FileNotFoundError):
            await provider.read_file_bytes("missing.bin")

        await provider.run_command("echo saved > out.txt; ln -s /etc/passwd leak.txt")
        saved = await provider.save_output_files(["out.txt", "leak.txt"], tmp_path)
        assert (tmp_path / "out.txt").read_text() == "saved\n"
        assert "outside the working directory" in saved.failed["leak.txt"]


async def test_allowlist_runs_literal_arguments() -> None:
    provider = make_provider(allowed_commands=[r"^echo", r"^if"])
    async with provider:
        assert (await provider.run_command("echo '$(whoami)'")).stdout == "$(whoami)\n"
        assert (await provider.run_command("if")).exit_code == 127
        assert (await provider.run_command("whoami")).error_kind == "command_not_allowed"


async def test_exit_deletes_the_machine() -> None:
    provider = make_provider()
    async with provider:
        machine_id = provider.machine_id
    assert machine_id is not None

    async with AsyncBoxd(region=os.environ.get("BOXD_E2E_REGION", "nyc")) as client:
        with pytest.raises(NotFoundError):
            for _ in range(15):
                await client.machines.get(machine_id)
                await anyio.sleep(1)


async def test_wrong_api_key_is_a_clear_error() -> None:
    provider = BoxdCodeExecToolProvider(api_key="not-a-real-key", region=os.environ.get("BOXD_E2E_REGION", "nyc"))
    with pytest.raises(AuthenticationError):
        async with provider:
            pass
