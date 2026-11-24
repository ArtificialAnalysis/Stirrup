"""Tests for BoxdCodeExecToolProvider against a fake client."""

from __future__ import annotations

import io
import shlex
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from PIL import Image

pytest.importorskip("boxd")

from boxd.errors import BoxdError, NotFoundError

from stirrup.tools.code_backends.boxd import (
    BOXD_WORKING_DIR,
    BoxdCodeExecToolProvider,
    _quote_argv_for_shell,
)

MACHINE_ID = "machine-id-1"


def exec_result(stdout: str = "", stderr: str = "", exit_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)


@pytest.fixture(autouse=True)
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOXD_API_KEY", "test-key")
    monkeypatch.delenv("BOXD_TOKEN", raising=False)
    monkeypatch.delenv("BOXD_ORG", raising=False)


@pytest.fixture
def client() -> MagicMock:
    fake = MagicMock()
    fake.close = AsyncMock()
    fake.machines.create = AsyncMock(return_value=SimpleNamespace(id=MACHINE_ID))
    fake.machines.wait_until_ready = AsyncMock()
    fake.machines.delete = AsyncMock()
    fake.machines.get = AsyncMock(side_effect=NotFoundError("no such machine"))
    fake.machines.exec = AsyncMock(return_value=exec_result())
    fake.machines.files.upload = AsyncMock(return_value=1)
    fake.machines.files.download = AsyncMock(return_value=b"file content")
    return fake


@pytest.fixture
def patched(client: MagicMock) -> Iterator[MagicMock]:
    with patch("stirrup.tools.code_backends.boxd.AsyncBoxd", return_value=client) as constructor:
        yield constructor


@pytest.mark.boxd
@pytest.mark.usefixtures("patched")
class TestBoxdLifecycle:
    async def test_machine_is_always_isolated(self, client: MagicMock) -> None:
        async with BoxdCodeExecToolProvider():
            pass
        assert client.machines.create.call_args.kwargs["isolated"] is True

    async def test_waits_for_ready_and_creates_workspace(self, client: MagicMock) -> None:
        async with BoxdCodeExecToolProvider():
            client.machines.wait_until_ready.assert_awaited_once_with(MACHINE_ID)
            assert client.machines.exec.call_args.args == (MACHINE_ID, ["mkdir", "-p", BOXD_WORKING_DIR])

    async def test_exit_deletes_by_id_and_closes(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            assert provider.machine_id == MACHINE_ID
        client.machines.delete.assert_awaited_once_with(MACHINE_ID)
        client.close.assert_awaited_once()
        assert provider.machine_id is None

    async def test_failed_setup_deletes_the_machine(self, client: MagicMock) -> None:
        client.machines.wait_until_ready.side_effect = BoxdError("not ready")
        with pytest.raises(BoxdError):
            async with BoxdCodeExecToolProvider():
                pass
        client.machines.delete.assert_awaited_once_with(MACHINE_ID)
        client.close.assert_awaited_once()

    async def test_failed_create_deletes_nothing(self, client: MagicMock) -> None:
        client.machines.create.side_effect = BoxdError("denied")
        with pytest.raises(BoxdError):
            async with BoxdCodeExecToolProvider():
                pass
        client.machines.delete.assert_not_awaited()
        client.close.assert_awaited_once()

    async def test_lost_create_response_deletes_by_looked_up_id(self, client: MagicMock) -> None:
        async def create(name: str, **_: object) -> None:
            client.machines.get.side_effect = None
            client.machines.get.return_value = SimpleNamespace(id="orphan-id", name=name)
            raise BoxdError("connection reset")

        client.machines.create.side_effect = create
        with pytest.raises(BoxdError, match="connection reset"):
            async with BoxdCodeExecToolProvider():
                pass
        client.machines.delete.assert_awaited_once_with("orphan-id")

    async def test_lookup_never_deletes_a_differently_named_machine(self, client: MagicMock) -> None:
        client.machines.create.side_effect = BoxdError("connection reset")
        client.machines.get.side_effect = None
        client.machines.get.return_value = SimpleNamespace(id="other-id", name="someone-else")
        with pytest.raises(BoxdError):
            async with BoxdCodeExecToolProvider():
                pass
        client.machines.delete.assert_not_awaited()

    async def test_second_enter_is_refused(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            with pytest.raises(RuntimeError, match="already has a running machine"):
                await provider.__aenter__()
        client.machines.create.assert_awaited_once()
        client.machines.delete.assert_awaited_once_with(MACHINE_ID)

    async def test_unexpected_delete_error_does_not_mask_session_error(self, client: MagicMock) -> None:
        client.machines.delete.side_effect = OSError("socket closed")
        with pytest.raises(KeyError):
            async with BoxdCodeExecToolProvider():
                raise KeyError("session error")

    async def test_cancelled_session_still_deletes(self, client: MagicMock) -> None:
        deleted: list[str] = []

        async def slow_delete(machine_id: str) -> None:
            await anyio.sleep(0.01)
            deleted.append(machine_id)

        client.machines.delete.side_effect = slow_delete
        with anyio.move_on_after(0.01):
            async with BoxdCodeExecToolProvider():
                await anyio.sleep(5)
        assert deleted == [MACHINE_ID]

    async def test_delete_failure_does_not_mask_session_error(self, client: MagicMock) -> None:
        client.machines.delete.side_effect = BoxdError("delete failed")
        with pytest.raises(KeyError):
            async with BoxdCodeExecToolProvider():
                raise KeyError("session error")
        client.close.assert_awaited_once()

    async def test_options_reach_create(self, client: MagicMock, patched: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider(
            api_key="key",
            org="my-org",
            region="fra",
            vcpu=2,
            memory="8G",
            labels={"team": "research"},
            name_prefix="run-",
        )
        async with provider:
            pass
        patched.assert_called_once_with(api_key="key", region="fra")
        args, kwargs = client.machines.create.call_args
        assert args[0].startswith("run-")
        assert kwargs["org"] == "my-org"
        assert kwargs["region"] == "fra"
        assert (kwargs["vcpu"], kwargs["memory"]) == (2, "8G")
        assert kwargs["labels"] == {"team": "research"}
        assert kwargs["auto_suspend_timeout"] == 0

    async def test_org_defaults_to_env(self, client: MagicMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("BOXD_ORG", "env-org")
        async with BoxdCodeExecToolProvider():
            pass
        assert client.machines.create.call_args.kwargs["org"] == "env-org"

    async def test_missing_credentials_is_a_clear_error(
        self, client: MagicMock, patched: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("BOXD_API_KEY")
        with pytest.raises(ValueError, match="BOXD_API_KEY"):
            async with BoxdCodeExecToolProvider():
                pass
        patched.assert_not_called()
        client.machines.create.assert_not_awaited()

    async def test_methods_require_started(self) -> None:
        provider = BoxdCodeExecToolProvider()
        with pytest.raises(RuntimeError, match="not started"):
            await provider.run_command("echo hi")
        with pytest.raises(RuntimeError, match="not started"):
            await provider.read_file_bytes("a.txt")


@pytest.mark.boxd
@pytest.mark.usefixtures("patched")
class TestBoxdRunCommand:
    async def test_runs_through_bash_in_workspace(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider(env={"KEY": "value"}, shell_timeout=42)
        async with provider:
            client.machines.exec.return_value = exec_result("out", "err", 7)
            result = await provider.run_command("[[ a == a ]] && echo out")

        call = client.machines.exec.await_args_list[-1]
        assert call.args == (MACHINE_ID, ["bash", "-c", "[[ a == a ]] && echo out"])
        assert call.kwargs == {"env": {"KEY": "value"}, "cwd": BOXD_WORKING_DIR, "timeout": 42}
        assert (result.exit_code, result.stdout, result.stderr) == (7, "out", "err")

    async def test_explicit_timeout_wins(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider(shell_timeout=42)
        async with provider:
            await provider.run_command("true", timeout=3)
        assert client.machines.exec.await_args_list[-1].kwargs["timeout"] == 3

    async def test_timeout_becomes_error_kind(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            client.machines.exec.side_effect = BoxdError("exec timed out after 3s")
            result = await provider.run_command("sleep 100", timeout=3)
            client.machines.exec.side_effect = None
        assert result.error_kind == "timeout"

    async def test_other_errors_become_a_result(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            client.machines.exec.side_effect = BoxdError("connection lost")
            result = await provider.run_command("true")
            calls = client.machines.exec.await_count
            client.machines.exec.side_effect = None
        assert (result.error_kind, result.stderr) == ("execution_error", "connection lost")
        assert calls == 2  # setup plus one attempt, no retry

    async def test_allowlist_runs_quoted_argv_without_bash(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider(allowed_commands=[r"^echo"])
        async with provider:
            await provider.run_command("echo '$(whoami)'")
            assert client.machines.exec.await_args_list[-1].args[1] == "'echo' '$(whoami)'"

            calls = client.machines.exec.await_count
            result = await provider.run_command("rm -rf /")
            assert result.error_kind == "command_not_allowed"
            assert client.machines.exec.await_count == calls

    @pytest.mark.parametrize("argv", [["echo", "$(whoami)"], ["echo", "it's"], ["echo", ""], ["if"]])
    def test_quote_argv_round_trips(self, argv: list[str]) -> None:
        assert shlex.split(_quote_argv_for_shell(argv)) == argv

    def test_quote_argv_quotes_reserved_command_word(self) -> None:
        assert _quote_argv_for_shell(["if"]) == "'if'"


@pytest.mark.boxd
@pytest.mark.usefixtures("patched")
class TestBoxdFiles:
    async def test_relative_paths_anchor_at_workspace(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            await provider.write_file_bytes("a/b.txt", b"x")
            await provider.read_file_bytes("a/b.txt")
        target = f"{BOXD_WORKING_DIR}/a/b.txt"
        client.machines.files.upload.assert_awaited_once_with(MACHINE_ID, target, b"x")
        client.machines.files.download.assert_awaited_once_with(MACHINE_ID, target)

    async def test_missing_file_raises_file_not_found(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            client.machines.files.download.side_effect = NotFoundError("no such file")
            with pytest.raises(FileNotFoundError):
                await provider.read_file_bytes("missing.txt")

    async def test_list_files_splits_nul(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            client.machines.exec.side_effect = [exec_result(), exec_result("a.txt\0sub/with space.txt\0")]
            assert await provider.list_files("data") == ["a.txt", "sub/with space.txt"]
            client.machines.exec.side_effect = None

    async def test_list_files_on_non_directory(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            client.machines.exec.return_value = exec_result(exit_code=1)
            assert await provider.list_files("a.txt") == []

    async def test_save_output_rejects_symlink_escape(self, client: MagicMock, temp_output_dir: Path) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            client.machines.exec.return_value = exec_result("/etc/passwd\0")
            result = await provider.save_output_files(["leak.txt"], temp_output_dir)
        assert "outside the working directory" in result.failed["leak.txt"]
        assert list(temp_output_dir.iterdir()) == []

    async def test_save_output_rejects_directory(self, client: MagicMock, temp_output_dir: Path) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            client.machines.exec.side_effect = [exec_result(f"{BOXD_WORKING_DIR}/data\0"), exec_result(exit_code=1)]
            result = await provider.save_output_files(["data"], temp_output_dir)
            client.machines.exec.side_effect = None
        assert "not a file" in result.failed["data"]

    async def test_save_output_preserves_relative_path(self, client: MagicMock, temp_output_dir: Path) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            client.machines.exec.side_effect = [exec_result(f"{BOXD_WORKING_DIR}/sub/out.txt\0"), exec_result()]
            result = await provider.save_output_files([f"{BOXD_WORKING_DIR}/sub/out.txt"], temp_output_dir)
            client.machines.exec.side_effect = None
        assert (temp_output_dir / "sub" / "out.txt").read_bytes() == b"file content"
        assert len(result.saved) == 1

    async def test_destination_escape_rejected(self, client: MagicMock) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            client.machines.exec.return_value = exec_result(f"{BOXD_WORKING_DIR}/out\0/etc/x\0")
            with pytest.raises(ValueError, match="escapes"):
                await provider.output_destination_identity("out/link/x", "out")

    async def test_upload_directory_keeps_its_name(self, client: MagicMock, sample_dir: Path) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            result = await provider.upload_files(sample_dir)
        assert len(result.uploaded) == 3
        for call in client.machines.files.upload.await_args_list:
            assert call.args[1].startswith(f"{BOXD_WORKING_DIR}/{sample_dir.name}/")

    async def test_upload_file_lands_in_workspace(self, client: MagicMock, sample_file: Path) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            result = await provider.upload_files(sample_file)
        assert result.failed == {}
        assert client.machines.files.upload.await_args.args[1] == f"{BOXD_WORKING_DIR}/{sample_file.name}"

    async def test_view_image_reports_a_damaged_png(self, client: MagicMock) -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(buffer, format="PNG")
        damaged = bytearray(buffer.getvalue())
        damaged[damaged.index(b"IDAT") + 5] ^= 0xFF
        client.machines.files.download.return_value = bytes(damaged)

        provider = BoxdCodeExecToolProvider()
        async with provider:
            with pytest.raises(ValueError, match="corrupt"):
                await provider.view_image("broken.png")

    async def test_view_image_read_error_is_a_value_error(self, client: MagicMock) -> None:
        client.machines.files.download.side_effect = BoxdError("is a directory")
        provider = BoxdCodeExecToolProvider()
        async with provider:
            with pytest.raises(ValueError, match="Could not read image"):
                await provider.view_image("plots.png")

    async def test_view_image_missing_file(self, client: MagicMock) -> None:
        client.machines.files.download.side_effect = NotFoundError("no such file")
        provider = BoxdCodeExecToolProvider()
        async with provider:
            with pytest.raises(FileNotFoundError):
                await provider.view_image("missing.png")

    async def test_view_image_rejects_other_types(self) -> None:
        provider = BoxdCodeExecToolProvider()
        async with provider:
            with pytest.raises(ValueError, match="Unsupported image type"):
                await provider.view_image("notes.txt")
