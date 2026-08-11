"""Tests for DaytonaCodeExecToolProvider backend."""

from __future__ import annotations

import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests in this module if daytona package is not installed
pytest.importorskip("daytona")

from daytona import DaytonaNotFoundError, DaytonaTimeoutError, DaytonaValidationError

from stirrup.tools.code_backends.base import OutputSourceRoot
from stirrup.tools.code_backends.daytona import DaytonaCodeExecToolProvider, _quote_argv_for_shell

WORK_DIR = "/home/daytona"


def _session_response(exit_code: int = 0, stdout: str = "hello world", stderr: str = "") -> MagicMock:
    response = MagicMock()
    response.exit_code = exit_code
    response.stdout = stdout
    response.stderr = stderr
    return response


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock Daytona AsyncSandbox."""
    sandbox = MagicMock()
    sandbox.delete = AsyncMock()
    sandbox.get_work_dir = AsyncMock(return_value=WORK_DIR)

    # Mock process/session API
    sandbox.process = MagicMock()
    sandbox.process.create_session = AsyncMock()
    sandbox.process.execute_session_command = AsyncMock(return_value=_session_response())
    sandbox.process.delete_session = AsyncMock()

    # Mock fs API
    sandbox.fs = MagicMock()
    mock_file_info = MagicMock()
    mock_file_info.is_dir = False
    sandbox.fs.get_file_info = AsyncMock(return_value=mock_file_info)
    sandbox.fs.upload_file = AsyncMock()
    sandbox.fs.upload_files = AsyncMock()
    sandbox.fs.download_file = AsyncMock(return_value=b"file content")

    return sandbox


@pytest.fixture
def mock_daytona(mock_sandbox: MagicMock) -> MagicMock:
    """Create a mock AsyncDaytona client that returns the mock sandbox."""
    client = MagicMock()
    client.create = AsyncMock(return_value=mock_sandbox)
    client.close = AsyncMock()
    return client


@pytest.mark.daytona
class TestDaytonaCodeExecToolProvider:
    """Tests for DaytonaCodeExecToolProvider."""

    async def test_create_and_cleanup(self, mock_daytona: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test sandbox creation and cleanup."""
        provider = DaytonaCodeExecToolProvider(timeout=300)

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                assert provider._sbx is not None  # noqa: SLF001
                assert provider.output_source_roots() == (OutputSourceRoot(path=WORK_DIR),)

            # Verify sandbox was deleted and the client closed on cleanup
            mock_sandbox.delete.assert_called_once()
            mock_daytona.close.assert_called_once()
            assert provider.output_source_roots() == ()

    async def test_lifetime_mapped_to_ttl_minutes(self, mock_daytona: MagicMock) -> None:
        """`timeout` seconds become a whole-minute wall-clock TTL."""
        provider = DaytonaCodeExecToolProvider(timeout=90, snapshot="my-snapshot")

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                params = mock_daytona.create.call_args.args[0]
                assert params.ttl_minutes == 2
                assert params.auto_stop_interval == 0
                assert params.snapshot == "my-snapshot"

    async def test_run_command(self, mock_daytona: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test basic command execution via a per-command session."""
        provider = DaytonaCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                result = await provider.run_command("echo 'hello world'")

                assert result.exit_code == 0
                assert result.stdout == "hello world"
                assert result.stderr == ""
                assert result.error_kind is None

                mock_sandbox.process.create_session.assert_called_once()
                request = mock_sandbox.process.execute_session_command.call_args.args[1]
                assert request.command == "( echo 'hello world'\n)"
                mock_sandbox.process.delete_session.assert_called_once()

    async def test_run_command_nonzero_exit(self, mock_daytona: MagicMock, mock_sandbox: MagicMock) -> None:
        """Non-zero exits are returned as results, with stdout and stderr split."""
        provider = DaytonaCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                mock_sandbox.process.execute_session_command = AsyncMock(
                    return_value=_session_response(exit_code=3, stdout="out", stderr="boom")
                )
                result = await provider.run_command("false")

                assert result.exit_code == 3
                assert result.stdout == "out"
                assert result.stderr == "boom"
                assert result.error_kind is None

    async def test_run_command_exceptions(self, mock_daytona: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test handling of Daytona-specific exceptions."""
        provider = DaytonaCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                # Test DaytonaValidationError (e.g. NUL bytes in the command)
                mock_sandbox.process.execute_session_command = AsyncMock(side_effect=DaytonaValidationError("invalid"))
                result = await provider.run_command("bad\x00command")
                assert result.error_kind == "invalid_argument"
                # The session is still torn down after a failure
                mock_sandbox.process.delete_session.assert_called_once()

                # Test DaytonaTimeoutError
                mock_sandbox.process.delete_session.reset_mock()
                mock_sandbox.process.execute_session_command = AsyncMock(side_effect=DaytonaTimeoutError("timeout"))
                result = await provider.run_command("sleep 1000")
                assert result.error_kind == "timeout"
                mock_sandbox.process.delete_session.assert_called_once()

    async def test_run_command_allowlist(self, mock_daytona: MagicMock, mock_sandbox: MagicMock) -> None:
        """Test command allowlist enforcement."""
        provider = DaytonaCodeExecToolProvider(allowed_commands=[r"^echo", r"^python"])

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                # Allowed command: re-quoted from parsed argv, with the command
                # word force-quoted so the sandbox shell cannot treat it as a
                # keyword or assignment.
                result = await provider.run_command("echo 'allowed'")
                assert result.error_kind is None
                request = mock_sandbox.process.execute_session_command.call_args.args[1]
                assert request.command == "( 'echo' allowed\n)"

                # Disallowed command never reaches the sandbox
                mock_sandbox.process.create_session.reset_mock()
                result = await provider.run_command("rm -rf /")
                assert result.error_kind == "command_not_allowed"
                mock_sandbox.process.create_session.assert_not_called()

    @pytest.mark.parametrize(
        "argv",
        [
            ["echo", "$(whoami)"],
            ["echo", "line1\nline2"],
            ["echo", "it's"],
            ["echo", ""],
            ["ls", "-n"],
        ],
    )
    def test_quote_argv_for_shell_round_trips(self, argv: list[str]) -> None:
        # Daytona hands a string to the session shell, so the re-quoting is
        # the whole guarantee: it must parse back to exactly the arguments
        # the allowlist vetted.
        assert shlex.split(_quote_argv_for_shell(argv)) == argv

    async def test_save_output_files(
        self, mock_daytona: MagicMock, mock_sandbox: MagicMock, temp_output_dir: Path
    ) -> None:
        """Test saving files from the Daytona sandbox."""
        provider = DaytonaCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                mock_sandbox.process.execute_session_command = AsyncMock(
                    return_value=_session_response(stdout=f"{WORK_DIR}/output.txt\n")
                )
                result = await provider.save_output_files([f"{WORK_DIR}/output.txt"], temp_output_dir)

                assert len(result.saved) == 1
                assert result.saved[0].source_path == f"{WORK_DIR}/output.txt"
                mock_sandbox.fs.get_file_info.assert_called()
                mock_sandbox.fs.download_file.assert_called()

    async def test_save_output_files_rejects_invalid_sources(
        self, mock_daytona: MagicMock, mock_sandbox: MagicMock, temp_output_dir: Path
    ) -> None:
        """Each resolve_output_source rejection path lands in `failed`."""
        provider = DaytonaCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                # realpath resolves outside the working root (e.g. a symlink)
                mock_sandbox.process.execute_session_command = AsyncMock(
                    return_value=_session_response(stdout="/etc/passwd\n")
                )
                result = await provider.save_output_files([f"{WORK_DIR}/leak.txt"], temp_output_dir)
                assert "outside Daytona execution root" in result.failed[f"{WORK_DIR}/leak.txt"]

                # realpath exits nonzero (missing or unreadable file)
                mock_sandbox.process.execute_session_command = AsyncMock(
                    return_value=_session_response(exit_code=1, stdout="")
                )
                result = await provider.save_output_files(["missing.txt"], temp_output_dir)
                assert "Could not resolve" in result.failed["missing.txt"]

                # realpath output malformed (no NUL-terminated path)
                mock_sandbox.process.execute_session_command = AsyncMock(return_value=_session_response(stdout=""))
                result = await provider.save_output_files(["weird.txt"], temp_output_dir)
                assert "Could not resolve" in result.failed["weird.txt"]

                # resolves to a directory, not a file
                mock_sandbox.process.execute_session_command = AsyncMock(
                    return_value=_session_response(stdout=f"{WORK_DIR}/data\n")
                )
                directory_info = MagicMock()
                directory_info.is_dir = True
                mock_sandbox.fs.get_file_info = AsyncMock(return_value=directory_info)
                result = await provider.save_output_files(["data"], temp_output_dir)
                assert "not a file" in result.failed["data"]

                # nothing was written to the output directory
                assert list(temp_output_dir.iterdir()) == []

    async def test_upload_files(
        self, mock_daytona: MagicMock, mock_sandbox: MagicMock, sample_file: Path, sample_dir: Path
    ) -> None:
        """Test uploading files to the Daytona sandbox."""
        provider = DaytonaCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                # Upload single file lands in the sandbox working directory
                result = await provider.upload_files(sample_file)
                assert len(result.uploaded) == 1
                assert result.uploaded[0].dest_path == f"{WORK_DIR}/{sample_file.name}"
                mock_sandbox.fs.upload_files.assert_called_once()

                # Upload directory: 3 files sent in a single batched request
                mock_sandbox.fs.upload_files.reset_mock()
                result = await provider.upload_files(sample_dir)
                assert len(result.uploaded) == 3  # 3 files in sample_dir
                mock_sandbox.fs.upload_files.assert_called_once()
                batch = mock_sandbox.fs.upload_files.call_args.args[0]
                assert len(batch) == 3
                assert all(upload.destination.startswith(f"{WORK_DIR}/{sample_dir.name}/") for upload in batch)

    async def test_upload_files_batch_failure_marks_all_failed(
        self, mock_daytona: MagicMock, mock_sandbox: MagicMock, sample_dir: Path
    ) -> None:
        """A failed batch request reports every planned file as failed."""
        provider = DaytonaCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                mock_sandbox.fs.upload_files = AsyncMock(side_effect=DaytonaTimeoutError("upload timed out"))
                result = await provider.upload_files(sample_dir)
                assert not result.uploaded
                assert len(result.failed) == 3
                assert all("upload timed out" in reason for reason in result.failed.values())

    async def test_read_file_bytes_missing_file(self, mock_daytona: MagicMock, mock_sandbox: MagicMock) -> None:
        """A missing file surfaces as FileNotFoundError."""
        provider = DaytonaCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            async with provider as _:
                mock_sandbox.fs.get_file_info = AsyncMock(side_effect=DaytonaNotFoundError("missing"))
                with pytest.raises(FileNotFoundError):
                    await provider.read_file_bytes(f"{WORK_DIR}/nope.txt")

    async def test_sandbox_cleaned_up_when_setup_fails(self, mock_daytona: MagicMock, mock_sandbox: MagicMock) -> None:
        """A failure after creation deletes the sandbox and closes the client."""
        provider = DaytonaCodeExecToolProvider()
        mock_sandbox.get_work_dir = AsyncMock(side_effect=DaytonaTimeoutError("boom"))

        with patch(
            "stirrup.tools.code_backends.daytona.AsyncDaytona",
            new=MagicMock(return_value=mock_daytona),
        ):
            with pytest.raises(DaytonaTimeoutError):
                async with provider as _:
                    pass

            mock_sandbox.delete.assert_called_once()
            mock_daytona.close.assert_called_once()
