"""Tests for DockerCodeExecToolProvider backend."""

from __future__ import annotations

import shlex
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests in this module if docker package is not installed
pytest.importorskip("docker")

from stirrup.tools.code_backends.docker import (
    DEFAULT_WORKING_DIR,
    DockerCodeExecToolProvider,
)


@pytest.fixture
def mock_docker_client() -> MagicMock:
    """Create a mock Docker client."""
    client = MagicMock()

    # Mock images API
    client.images.get = MagicMock()
    client.images.pull = MagicMock()

    # Mock container
    mock_container = MagicMock()
    mock_container.short_id = "abc123"
    mock_container.stop = MagicMock()
    mock_container.remove = MagicMock()
    client.containers.run = MagicMock(return_value=mock_container)

    return client


@pytest.mark.docker
class TestDockerCodeExecToolProvider:
    """Tests for DockerCodeExecToolProvider."""

    async def test_from_image_and_lifecycle(self, mock_docker_client: MagicMock, tmp_path: Path) -> None:
        """Test factory method and container lifecycle."""
        provider = DockerCodeExecToolProvider.from_image(
            "python:3.12-slim",
            temp_base_dir=tmp_path,
        )

        with (
            patch("stirrup.tools.code_backends.docker.docker.from_env", return_value=mock_docker_client),
            patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread,
        ):
            # Make to_thread.run_sync execute the function directly
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args) if not args else fn)

            async with provider as _:
                # Verify temp directory created
                assert provider.temp_dir is not None
                assert provider.temp_dir.exists()

                # Verify container started
                assert provider.container_id == "abc123"
                mock_docker_client.containers.run.assert_called_once()

            # Verify cleanup called
            mock_docker_client.containers.run.return_value.stop.assert_called()
            mock_docker_client.containers.run.return_value.remove.assert_called()

    async def test_run_command(self, mock_docker_client: MagicMock, tmp_path: Path) -> None:
        """Test command execution in Docker container."""
        provider = DockerCodeExecToolProvider.from_image("python:3.12-slim", temp_base_dir=tmp_path)

        # Setup mock exec_run response
        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 0
        mock_exec_result.output = (b"hello world\n", b"")
        mock_docker_client.containers.run.return_value.exec_run = MagicMock(return_value=mock_exec_result)

        with (
            patch("stirrup.tools.code_backends.docker.docker.from_env", return_value=mock_docker_client),
            patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread,
        ):
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args) if not args else fn)

            async with provider as _:
                result = await provider.run_command("echo 'hello world'")

                assert result.exit_code == 0
                assert result.stdout == "hello world\n"
                assert result.stderr == ""
                assert result.error_kind is None

    async def test_run_command_wraps_with_timeout(self, mock_docker_client: MagicMock, tmp_path: Path) -> None:
        """Regression: every command must be wrapped with coreutils timeout(1).

        The wrapper is what prevents orphan descendants in the container when
        the client gives up — without it, backgrounded grandchildren reparent
        to PID 1 and leak (see moby/moby#9098). This test pins the exact form
        so the protection can't silently regress.
        """
        provider = DockerCodeExecToolProvider.from_image(
            "python:3.12-slim",
            temp_base_dir=tmp_path,
            shell_timeout=42,
        )

        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 0
        mock_exec_result.output = (b"", b"")
        exec_run = MagicMock(return_value=mock_exec_result)
        mock_docker_client.containers.run.return_value.exec_run = exec_run

        with (
            patch("stirrup.tools.code_backends.docker.docker.from_env", return_value=mock_docker_client),
            patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread,
        ):
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args) if not args else fn)

            async with provider as _:
                user_cmd = "echo 'hi' && sleep 0"
                await provider.run_command(user_cmd)
                # Capture inside the context — __aexit__ issues a chown via
                # exec_run which would otherwise overwrite call_args.
                cmd_kwarg = exec_run.call_args.kwargs["cmd"]

        expected_inner = f"timeout --kill-after=5s 42s bash -c {shlex.quote(user_cmd)}"
        assert cmd_kwarg == ["bash", "-c", expected_inner], (
            f"docker exec must wrap user commands with timeout(1); got {cmd_kwarg!r}"
        )

    async def test_run_command_timeout_exit_codes_map_to_timeout(
        self, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        """Exit 124 (SIGTERM expiry) and 137 (--kill-after SIGKILL) both surface as timeouts."""
        provider = DockerCodeExecToolProvider.from_image("python:3.12-slim", temp_base_dir=tmp_path)

        for exit_code in (124, 137):
            mock_exec_result = MagicMock()
            mock_exec_result.exit_code = exit_code
            mock_exec_result.output = (b"partial-stdout", b"partial-stderr")
            mock_docker_client.containers.run.return_value.exec_run = MagicMock(return_value=mock_exec_result)

            with (
                patch("stirrup.tools.code_backends.docker.docker.from_env", return_value=mock_docker_client),
                patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread,
            ):
                mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args) if not args else fn)

                async with provider as _:
                    result = await provider.run_command("sleep 999", timeout=1)

            assert result.error_kind == "timeout", f"exit {exit_code} should map to timeout"
            assert result.exit_code == exit_code
            assert "partial-stdout" in result.stdout
            assert "partial-stderr" in result.stderr
            assert "timed out" in result.stderr.lower()

    async def test_run_command_missing_timeout_binary_advice(
        self, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        """If the image lacks coreutils timeout(1), exit 127 should surface a clear pointer."""
        provider = DockerCodeExecToolProvider.from_image("python:3.12-slim", temp_base_dir=tmp_path)

        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 127
        mock_exec_result.output = (b"", b"bash: line 1: timeout: command not found\n")
        mock_docker_client.containers.run.return_value.exec_run = MagicMock(return_value=mock_exec_result)

        with (
            patch("stirrup.tools.code_backends.docker.docker.from_env", return_value=mock_docker_client),
            patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread,
        ):
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args) if not args else fn)

            async with provider as _:
                result = await provider.run_command("echo hi")

        assert result.error_kind == "image_missing_timeout"
        assert result.advice is not None and "coreutils" in result.advice

    async def test_run_command_allowlist(self, mock_docker_client: MagicMock, tmp_path: Path) -> None:
        """Test command allowlist enforcement."""
        provider = DockerCodeExecToolProvider.from_image(
            "python:3.12-slim",
            allowed_commands=[r"^echo", r"^python"],
            temp_base_dir=tmp_path,
        )

        mock_exec_result = MagicMock()
        mock_exec_result.exit_code = 0
        mock_exec_result.output = (b"allowed\n", b"")
        mock_docker_client.containers.run.return_value.exec_run = MagicMock(return_value=mock_exec_result)

        with (
            patch("stirrup.tools.code_backends.docker.docker.from_env", return_value=mock_docker_client),
            patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread,
        ):
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args) if not args else fn)

            async with provider as _:
                # Allowed command
                result = await provider.run_command("echo 'allowed'")
                assert result.error_kind is None

                # Disallowed command
                result = await provider.run_command("rm -rf /")
                assert result.error_kind == "command_not_allowed"

    async def test_save_output_files(
        self, mock_docker_client: MagicMock, tmp_path: Path, temp_output_dir: Path
    ) -> None:
        """Test saving files from container (via mounted volume)."""
        provider = DockerCodeExecToolProvider.from_image("python:3.12-slim", temp_base_dir=tmp_path)

        with (
            patch("stirrup.tools.code_backends.docker.docker.from_env", return_value=mock_docker_client),
            patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread,
        ):
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args) if not args else fn)

            async with provider as _:
                # Create a file in the temp dir (simulating container creating it)
                test_file = provider.temp_dir / "output.txt"
                test_file.write_text("test content")

                # Test relative path
                result = await provider.save_output_files(["output.txt"], temp_output_dir)
                assert len(result.saved) == 1
                assert (temp_output_dir / "output.txt").read_text() == "test content"

                # Create another file for absolute path test
                test_file2 = provider.temp_dir / "output2.txt"
                test_file2.write_text("test content 2")

                # Test absolute container path (e.g., /workspace/output2.txt)
                abs_path = f"{DEFAULT_WORKING_DIR}/output2.txt"
                result = await provider.save_output_files([abs_path], temp_output_dir)
                assert len(result.saved) == 1

    async def test_upload_files(
        self, mock_docker_client: MagicMock, tmp_path: Path, sample_file: Path, sample_dir: Path
    ) -> None:
        """Test uploading files to container (via mounted volume)."""
        provider = DockerCodeExecToolProvider.from_image("python:3.12-slim", temp_base_dir=tmp_path)

        with (
            patch("stirrup.tools.code_backends.docker.docker.from_env", return_value=mock_docker_client),
            patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread,
        ):
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args) if not args else fn)

            async with provider as _:
                # Upload single file
                result = await provider.upload_files(sample_file)
                assert len(result.uploaded) == 1
                assert (provider.temp_dir / sample_file.name).exists()

                # Upload directory
                result = await provider.upload_files(sample_dir)
                assert len(result.uploaded) == 3
                assert (provider.temp_dir / sample_dir.name / "file1.txt").exists()
