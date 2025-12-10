"""Tests for E2BCodeExecToolProvider backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests in this module if e2b package is not installed
pytest.importorskip("e2b_code_interpreter")

from e2b import InvalidArgumentException, TimeoutException
from e2b.sandbox.filesystem.filesystem import FileType
from e2b_code_interpreter import CommandExitException

from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider


@pytest.fixture
def mock_sandbox() -> MagicMock:
    """Create a mock E2B AsyncSandbox."""
    sandbox = MagicMock()
    sandbox.kill = AsyncMock()

    # Mock commands API
    sandbox.commands = MagicMock()
    mock_result = MagicMock()
    mock_result.exit_code = 0
    mock_result.stdout = "hello world"
    mock_result.stderr = ""
    sandbox.commands.run = AsyncMock(return_value=mock_result)

    # Mock files API
    sandbox.files = MagicMock()
    sandbox.files.exists = AsyncMock(return_value=True)
    sandbox.files.write = AsyncMock()
    sandbox.files.read = AsyncMock(return_value=b"file content")

    mock_file_info = MagicMock()
    mock_file_info.type = FileType.FILE
    sandbox.files.get_info = AsyncMock(return_value=mock_file_info)

    return sandbox


@pytest.mark.e2b
class TestE2BCodeExecToolProvider:
    """Tests for E2BCodeExecToolProvider."""

    async def test_create_and_cleanup(self, mock_sandbox: MagicMock) -> None:
        """Test sandbox creation and cleanup."""
        provider = E2BCodeExecToolProvider(timeout=300)

        with patch(
            "stirrup.tools.code_backends.e2b.AsyncSandbox.create",
            new=AsyncMock(return_value=mock_sandbox),
        ):
            async with provider as _:
                assert provider._sbx is not None  # noqa: SLF001

            # Verify sandbox was killed on cleanup
            mock_sandbox.kill.assert_called_once()

    async def test_run_command(self, mock_sandbox: MagicMock) -> None:
        """Test basic command execution."""
        provider = E2BCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.e2b.AsyncSandbox.create",
            new=AsyncMock(return_value=mock_sandbox),
        ):
            async with provider as _:
                result = await provider.run_command("echo 'hello world'")

                assert result.exit_code == 0
                assert result.stdout == "hello world"
                assert result.stderr == ""
                assert result.error_kind is None
                mock_sandbox.commands.run.assert_called()

    async def test_run_command_exceptions(self, mock_sandbox: MagicMock) -> None:
        """Test handling of E2B-specific exceptions."""
        provider = E2BCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.e2b.AsyncSandbox.create",
            new=AsyncMock(return_value=mock_sandbox),
        ):
            async with provider as _:
                # Test CommandExitException (non-zero exit)
                mock_sandbox.commands.run = AsyncMock(
                    side_effect=CommandExitException(stderr="error", stdout="out", exit_code=1, error=None)
                )
                result = await provider.run_command("false")
                assert result.exit_code == 1
                assert result.stdout == "out"
                assert result.stderr == "error"

                # Test InvalidArgumentException
                mock_sandbox.commands.run = AsyncMock(side_effect=InvalidArgumentException("invalid"))
                result = await provider.run_command("bad\x00command")
                assert result.error_kind == "invalid_argument"

                # Test TimeoutException
                mock_sandbox.commands.run = AsyncMock(side_effect=TimeoutException("timeout"))
                result = await provider.run_command("sleep 1000")
                assert result.error_kind == "timeout"

    async def test_run_command_allowlist(self, mock_sandbox: MagicMock) -> None:
        """Test command allowlist enforcement."""
        provider = E2BCodeExecToolProvider(allowed_commands=[r"^echo", r"^python"])

        with patch(
            "stirrup.tools.code_backends.e2b.AsyncSandbox.create",
            new=AsyncMock(return_value=mock_sandbox),
        ):
            async with provider as _:
                # Allowed command
                result = await provider.run_command("echo 'allowed'")
                assert result.error_kind is None

                # Disallowed command
                result = await provider.run_command("rm -rf /")
                assert result.error_kind == "command_not_allowed"

    async def test_save_output_files(self, mock_sandbox: MagicMock, temp_output_dir: Path) -> None:
        """Test saving files from E2B sandbox."""
        provider = E2BCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.e2b.AsyncSandbox.create",
            new=AsyncMock(return_value=mock_sandbox),
        ):
            async with provider as _:
                result = await provider.save_output_files(["/home/user/output.txt"], temp_output_dir)

                assert len(result.saved) == 1
                assert result.saved[0].source_path == "/home/user/output.txt"
                mock_sandbox.files.exists.assert_called()
                mock_sandbox.files.read.assert_called()

                # Test non-existent file
                mock_sandbox.files.exists = AsyncMock(return_value=False)
                result = await provider.save_output_files(["/nonexistent.txt"], temp_output_dir)
                assert len(result.failed) == 1

    async def test_upload_files(self, mock_sandbox: MagicMock, sample_file: Path, sample_dir: Path) -> None:
        """Test uploading files to E2B sandbox."""
        provider = E2BCodeExecToolProvider()

        with patch(
            "stirrup.tools.code_backends.e2b.AsyncSandbox.create",
            new=AsyncMock(return_value=mock_sandbox),
        ):
            async with provider as _:
                # Upload single file
                result = await provider.upload_files(sample_file)
                assert len(result.uploaded) == 1
                mock_sandbox.files.write.assert_called()

                # Upload directory
                mock_sandbox.files.write.reset_mock()
                result = await provider.upload_files(sample_dir)
                assert len(result.uploaded) == 3  # 3 files in sample_dir
                assert mock_sandbox.files.write.call_count == 3
