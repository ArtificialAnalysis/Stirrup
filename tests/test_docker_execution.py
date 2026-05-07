"""Tests for DockerCodeExecToolProvider backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip all tests in this module if docker package is not installed
pytest.importorskip("docker")

from docker.errors import APIError, ImageNotFound

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

    @pytest.fixture(autouse=True)
    def _isolate_image_prep_locks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reset the class-level lock dict so tests don't inherit each other's state."""
        monkeypatch.setattr(DockerCodeExecToolProvider, "_image_prep_locks", {})

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

    async def test_prepare_image_recovers_from_concurrent_pull_race(
        self, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        """Transient ImageNotFound followed by a failed pull must not raise if
        a retry of images.get succeeds (i.e. a sibling task finished preparing
        the same image in the meantime)."""
        provider = DockerCodeExecToolProvider.from_image(
            "concurrent-race-test:local",
            temp_base_dir=tmp_path,
        )

        # First get() raises ImageNotFound (sibling task is mid-create and
        # the daemon returned a transient 404); pull() raises APIError (no
        # registry); retry get() succeeds because the sibling task finished.
        mock_docker_client.images.get = MagicMock(
            side_effect=[ImageNotFound("not found"), MagicMock()],
        )
        mock_docker_client.images.pull = MagicMock(
            side_effect=APIError("404 Client Error: pull access denied"),
        )

        provider._client = mock_docker_client  # noqa: SLF001 — short-circuits docker.from_env
        with patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread:
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args))
            image_name = await provider._prepare_image()  # noqa: SLF001

        assert image_name == "concurrent-race-test:local"
        assert mock_docker_client.images.get.call_count == 2
        assert mock_docker_client.images.pull.call_count == 1

    async def test_prepare_image_still_raises_when_image_truly_missing(
        self, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        """If both the initial get() and the retry after failed pull() report
        ImageNotFound, the RuntimeError must still surface."""
        provider = DockerCodeExecToolProvider.from_image(
            "truly-missing-image:local",
            temp_base_dir=tmp_path,
        )

        mock_docker_client.images.get = MagicMock(side_effect=ImageNotFound("not found"))
        mock_docker_client.images.pull = MagicMock(
            side_effect=APIError("404 Client Error: pull access denied"),
        )

        provider._client = mock_docker_client  # noqa: SLF001 — short-circuits docker.from_env
        with patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread:
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args))
            with pytest.raises(RuntimeError, match="Failed to pull Docker image"):
                await provider._prepare_image()  # noqa: SLF001

    async def test_prepare_image_surfaces_original_pull_error_when_retry_get_also_fails(
        self, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        """If the retry images.get() raises a non-ImageNotFound error (e.g. a
        transient daemon hiccup), the original pull failure must still
        surface as RuntimeError rather than being masked by the retry error."""
        provider = DockerCodeExecToolProvider.from_image(
            "retry-hiccup:local",
            temp_base_dir=tmp_path,
        )

        mock_docker_client.images.get = MagicMock(
            side_effect=[
                ImageNotFound("not found"),
                APIError("500 daemon hiccup on retry"),
            ],
        )
        mock_docker_client.images.pull = MagicMock(
            side_effect=APIError("404 Client Error: pull access denied"),
        )

        provider._client = mock_docker_client  # noqa: SLF001 — short-circuits docker.from_env
        with patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread:
            mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args))
            with pytest.raises(RuntimeError, match="Failed to pull Docker image"):
                await provider._prepare_image()  # noqa: SLF001

    def test_image_prep_lock_is_shared_per_image_name(self) -> None:
        """Providers targeting the same image name must share the same lock,
        but distinct image names must get distinct locks."""
        lock_a1 = DockerCodeExecToolProvider._get_image_lock("shared-name:tag")  # noqa: SLF001
        lock_a2 = DockerCodeExecToolProvider._get_image_lock("shared-name:tag")  # noqa: SLF001
        lock_b = DockerCodeExecToolProvider._get_image_lock("other-name:tag")  # noqa: SLF001

        assert lock_a1 is lock_a2
        assert lock_a1 is not lock_b
