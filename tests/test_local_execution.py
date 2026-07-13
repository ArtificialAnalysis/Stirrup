"""Tests for LocalCodeExecToolProvider backend."""

import os
import subprocess
from pathlib import Path

import anyio
import pytest
from anyio.to_thread import run_sync

from stirrup.tools.code_backends.local import LocalCodeExecToolProvider


def _bash_recursively_expands_array_index(probe_dir: Path) -> bool:
    marker = probe_dir / "array-expansion-marker"
    env = os.environ.copy()
    env["STIRRUP_TEST_INDEX"] = "$(touch array-expansion-marker)"
    subprocess.run(
        ["bash", "-c", 'printf %s "${VALUES[STIRRUP_TEST_INDEX]}"'],
        capture_output=True,
        text=True,
        check=False,
        cwd=probe_dir,
        env=env,
    )
    supported = marker.exists()
    marker.unlink(missing_ok=True)
    return supported


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

    async def test_run_command_timeout_kills_descendants(self) -> None:
        """Regression: a timed-out command must not leave orphaned grandchildren.

        Without start_new_session + killpg, bash's backgrounded descendants
        reparent to init and leak indefinitely. Uses a unique sleep duration
        as a marker so the check is robust to concurrent processes on the host.
        """
        marker = "919293"
        provider = LocalCodeExecToolProvider()
        async with provider as _:
            result = await provider.run_command(
                f"sleep {marker} & sleep {marker} & wait",
                timeout=1,
            )
            assert result.error_kind == "timeout"

        def list_survivors() -> list[str]:
            ps = subprocess.run(["ps", "-eo", "command"], capture_output=True, text=True, check=True)
            return [line for line in ps.stdout.splitlines() if f"sleep {marker}" in line]

        # Poll briefly for kernel to reap signalled descendants.
        survivors: list[str] = []
        for _ in range(10):
            await anyio.sleep(0.2)
            survivors = await run_sync(list_survivors)
            if not survivors:
                break
        assert not survivors, f"orphaned descendants after timeout: {survivors}"

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

    async def test_run_command_allowlist_does_not_execute_chained_command(self) -> None:
        provider = LocalCodeExecToolProvider(allowed_commands=[r"^echo"])

        async with provider:
            result = await provider.run_command("echo allowed; printf bypassed > forbidden.txt")
            forbidden_file_exists = await provider.file_exists("forbidden.txt")

        assert result.error_kind == "command_not_allowed"
        assert forbidden_file_exists is False

    async def test_run_command_allowlist_does_not_execute_ansi_c_quoted_array_assignment(
        self,
    ) -> None:
        provider = LocalCodeExecToolProvider(allowed_commands=[r"^echo"])

        async with provider:
            result = await provider.run_command(r"echo[$'a\'b']=x touch forbidden.txt")
            forbidden_file_exists = await provider.file_exists("forbidden.txt")

        assert result.error_kind == "command_not_allowed"
        assert forbidden_file_exists is False

    @pytest.mark.parametrize(
        ("allowed_pattern", "command"),
        [
            (r".*", "MODE=test echo allowed"),
            (r"^echo", "echo=x touch forbidden.txt"),
            (r"^echo", "echo[0]=x touch forbidden.txt"),
            (r"^echo", "echo[1 + 1]=x touch forbidden.txt"),
            (r"^echo", "echo allowed && printf bypassed"),
            (r"^echo", "echo allowed || printf bypassed"),
            (r"^echo", "echo allowed | cat"),
            (r"^echo", "echo allowed > output.txt"),
            (r"^echo", "echo allowed < input.txt"),
            (r"^echo", "echo allowed\nprintf bypassed"),
            (r"^echo", "echo $(printf bypassed)"),
            (r"^echo", "echo `printf bypassed`"),
            (r"^echo", 'echo "$STIRRUP_TEST_VALUE"'),
            (r"^echo", 'echo "${VALUES[STIRRUP_TEST_INDEX]}"'),
            (r"^echo", "echo $?"),
            (r"^echo", "echo $[VALUES[STIRRUP_TEST_INDEX]]"),
            (r"^echo", r"echo $'it\'s safe'; touch forbidden.txt"),
            (r"^echo", 'echo "it\'s safe"; touch forbidden.txt'),
            (r"^echo", "echo $\\\n(touch forbidden.txt)"),
            (r"echo.*", "bash -c 'echo allowed; touch forbidden.txt'"),
        ],
    )
    async def test_run_command_allowlist_rejects_shell_features(
        self,
        allowed_pattern: str,
        command: str,
    ) -> None:
        provider = LocalCodeExecToolProvider(allowed_commands=[allowed_pattern])

        async with provider:
            result = await provider.run_command(command)

        assert result.error_kind == "command_not_allowed"

    @pytest.mark.parametrize(
        "command",
        [
            'echo["1"]=x touch forbidden.txt',
            "echo['1']=x touch forbidden.txt",
            r"echo[1\+1]=x touch forbidden.txt",
        ],
    )
    async def test_run_command_allowlist_rejects_quoted_or_escaped_array_assignments(
        self,
        command: str,
    ) -> None:
        provider = LocalCodeExecToolProvider(allowed_commands=[r"^echo"])

        async with provider:
            result = await provider.run_command(command)
            side_effect_exists = await provider.file_exists("forbidden.txt")

        assert result.error_kind == "command_not_allowed"
        assert side_effect_exists is False

    @pytest.mark.parametrize(
        ("command", "expected_stdout"),
        [
            ("echo 'a;b && c || d | e > f < g\n$() `x`'", "a;b && c || d | e > f < g\n$() `x`\n"),
            ('echo "it\'s ; literal"', "it's ; literal\n"),
            (r"echo $'it\'s; literal'", "it's; literal\n"),
            (r"echo a\;b \$VALUE \${VALUE}", "a;b $VALUE ${VALUE}\n"),
            ("ec\\\nho allowed", "allowed\n"),
        ],
    )
    async def test_run_command_allowlist_allows_literal_shell_characters(
        self,
        command: str,
        expected_stdout: str,
    ) -> None:
        provider = LocalCodeExecToolProvider(allowed_commands=[r"^echo"])

        async with provider:
            result = await provider.run_command(command)

        assert result.error_kind is None
        assert result.stdout == expected_stdout

    async def test_run_command_allowlist_blocks_recursive_array_expansion_side_effect(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        if not await run_sync(_bash_recursively_expands_array_index, tmp_path):
            pytest.skip("installed Bash does not recursively expand array index values")

        monkeypatch.setenv("STIRRUP_TEST_INDEX", "$(touch forbidden.txt)")
        provider = LocalCodeExecToolProvider(allowed_commands=[r"^echo"])

        async with provider:
            result = await provider.run_command('echo "${VALUES[STIRRUP_TEST_INDEX]}"')
            side_effect_exists = await provider.file_exists("forbidden.txt")

        assert result.error_kind == "command_not_allowed"
        assert side_effect_exists is False

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
