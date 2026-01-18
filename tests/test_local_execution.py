"""Tests for LocalCodeExecToolProvider backend."""

from pathlib import Path

from stirrup.tools.code_backends.local import LocalCodeExecToolProvider


class TestLocalCodeExecToolProvider:
    """Tests for LocalCodeExecToolProvider."""

    async def test_create_and_cleanup(self) -> None:
        """Test that temp directory is created and cleaned up properly."""
        provider = LocalCodeExecToolProvider()

        # Before entering context, temp_dir should be None
        assert provider.temp_dir is None

        async with provider as _:
            # During context, temp_dir should exist
            assert provider.temp_dir is not None
            assert provider.temp_dir.exists()
            assert provider.temp_dir.is_dir()
            temp_dir_path = provider.temp_dir

        # After context exit, temp_dir should be cleaned up
        assert not temp_dir_path.exists()

    async def test_run_command(self) -> None:
        """Test basic command execution with stdout, stderr, and exit code capture."""
        provider = LocalCodeExecToolProvider()

        async with provider as _:
            # Test stdout capture
            result = await provider.run_command("echo 'hello world'")
            assert result.exit_code == 0
            assert "hello world" in result.stdout
            assert result.stderr == ""
            assert result.error_kind is None

            # Test stderr capture
            result = await provider.run_command("echo 'error message' >&2")
            assert result.exit_code == 0
            assert "error message" in result.stderr

            # Test non-zero exit code
            result = await provider.run_command("exit 42")
            assert result.exit_code == 42

    async def test_run_command_timeout(self) -> None:
        """Test that commands timeout correctly."""
        provider = LocalCodeExecToolProvider()

        async with provider as _:
            result = await provider.run_command("sleep 10", timeout=1)
            assert result.error_kind == "timeout"
            assert "timed out" in result.stderr.lower()

    async def test_run_command_allowlist(self) -> None:
        """Test command allowlist enforcement."""
        provider = LocalCodeExecToolProvider(allowed_commands=[r"^echo", r"^ls"])

        async with provider as _:
            # Allowed command should work
            result = await provider.run_command("echo 'allowed'")
            assert result.exit_code == 0
            assert result.error_kind is None

            # Disallowed command should be rejected
            result = await provider.run_command("cat /etc/passwd")
            assert result.error_kind == "command_not_allowed"
            assert "not allowed" in result.stderr.lower()

    async def test_save_output_files(self, temp_output_dir: Path) -> None:
        """Test saving files from the execution environment."""
        provider = LocalCodeExecToolProvider()

        async with provider as _:
            # Create a file in the temp directory
            await provider.run_command("echo 'test content' > output.txt")

            # Save the file
            result = await provider.save_output_files(["output.txt"], temp_output_dir)
            assert len(result.saved) == 1
            assert result.saved[0].source_path == "output.txt"
            assert result.saved[0].output_path == temp_output_dir / "output.txt"
            assert (temp_output_dir / "output.txt").read_text().strip() == "test content"

            # Original file should be moved (not exist in temp)
            assert provider.temp_dir is not None
            assert not (provider.temp_dir / "output.txt").exists()

            # Test failure case - non-existent file
            result = await provider.save_output_files(["nonexistent.txt"], temp_output_dir)
            assert len(result.failed) == 1
            assert "nonexistent.txt" in result.failed

    async def test_upload_files(self, sample_file: Path, sample_dir: Path) -> None:
        """Test uploading files to the execution environment."""
        provider = LocalCodeExecToolProvider()

        async with provider as _:
            assert provider.temp_dir is not None
            # Upload single file
            result = await provider.upload_files(sample_file)
            assert len(result.uploaded) == 1
            assert result.uploaded[0].source_path == sample_file
            uploaded_path = provider.temp_dir / sample_file.name
            assert uploaded_path.exists()
            assert uploaded_path.read_text() == "Hello, World!"

            # Upload directory
            result = await provider.upload_files(sample_dir)
            assert len(result.uploaded) == 3  # file1.txt, file2.txt, subdir/file3.txt
            assert (provider.temp_dir / sample_dir.name / "file1.txt").exists()
            assert (provider.temp_dir / sample_dir.name / "subdir" / "file3.txt").exists()

            # Test failure case - non-existent file
            result = await provider.upload_files(Path("/nonexistent/file.txt"))
            assert len(result.failed) == 1

    async def test_file_exists(self) -> None:
        """Test file_exists method."""
        provider = LocalCodeExecToolProvider()

        async with provider as _:
            # Create a file
            await provider.run_command("echo 'test' > exists.txt")

            # File should exist
            assert await provider.file_exists("exists.txt") is True

            # Non-existent file should return False
            assert await provider.file_exists("nonexistent.txt") is False

            # Directory should return False (only files)
            await provider.run_command("mkdir testdir")
            assert await provider.file_exists("testdir") is False
