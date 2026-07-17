"""Tests for LocalCodeExecToolProvider backend."""

import subprocess
from pathlib import Path

import anyio
import pytest
from anyio.to_thread import run_sync

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
            # Assignment prefixes.
            (r".*", "MODE=test echo allowed"),
            (r".*", "MODE+=test echo allowed"),
            (r"^echo", "echo=x touch forbidden.txt"),
            (r"^echo", "echo[0]=x touch forbidden.txt"),
            # Unquoted shell operators, spaced or attached to a word.
            (r"^echo", "echo allowed && printf bypassed"),
            (r"^echo", "echo allowed || printf bypassed"),
            (r"^echo", "echo allowed | cat"),
            (r"^echo", "echo allowed > output.txt"),
            (r"^echo", "echo allowed < input.txt"),
            (r"^echo", "echo allowed & printf bypassed"),
            (r"^echo", "echo allowed >output.txt"),
            (r"^echo", "echo allowed;printf bypassed"),
            (r"^echo", "echo allowed 2>&1"),
            (r"^echo", "echo allowed\nprintf bypassed"),
            (r"^printf", "printf 'a\nb'\nls"),
            # Unquoted substitution and subshell syntax.
            (r"^echo", "echo $(printf bypassed)"),
            (r"^echo", "echo `printf bypassed`"),
            (r"^echo", "echo <(printf bypassed)"),
            (r".*", "(touch forbidden.txt)"),
            # Bash-only quoting that the parser cannot represent fails closed.
            (r"^echo", r"echo $'it\'s; literal'"),
            # The pattern must match the executed command, not a quoted inner string.
            (r"echo.*", "bash -c 'echo allowed; touch forbidden.txt'"),
            # A prefix word shifts the executed command out from under the pattern.
            (r"^echo", "time echo allowed"),
            # Line continuations are not collapsed; the mangled word fails the pattern.
            (r"^echo", "ec\\\nho allowed"),
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
        "command",
        [
            "coproc touch forbidden.txt",
            "! touch forbidden.txt",
            "echo[1 + 1]=x touch forbidden.txt",
        ],
    )
    async def test_run_command_allowlist_shell_only_words_fail_without_side_effects(
        self,
        command: str,
    ) -> None:
        # Without a shell there are no keywords or compound assignments: the
        # first word is looked up as a binary, fails to exec, and nothing
        # else runs.
        provider = LocalCodeExecToolProvider(allowed_commands=[r".*"])

        async with provider:
            result = await provider.run_command(command)
            side_effect_exists = await provider.file_exists("forbidden.txt")

        assert result.exit_code != 0
        assert side_effect_exists is False

    @pytest.mark.parametrize(
        "command",
        [
            "echo 'a; touch forbidden.txt",
            'echo "a; touch forbidden.txt',
            r"echo $'a; touch forbidden.txt",
        ],
    )
    async def test_run_command_allowlist_rejects_unterminated_quotes(
        self,
        command: str,
    ) -> None:
        # An unterminated quote cannot be parsed, so it must fail closed.
        provider = LocalCodeExecToolProvider(allowed_commands=[r"^echo"])

        async with provider:
            result = await provider.run_command(command)
            side_effect_exists = await provider.file_exists("forbidden.txt")

        assert result.error_kind == "command_not_allowed"
        assert side_effect_exists is False

    async def test_run_command_allowlist_anchors_patterns_at_start(self) -> None:
        # Patterns match from the start of the command, so an unanchored pattern
        # must not match a command that merely contains it as a substring.
        provider = LocalCodeExecToolProvider(allowed_commands=[r"echo"])

        async with provider:
            result = await provider.run_command("xecho allowed")

        assert result.error_kind == "command_not_allowed"

    async def test_run_command_allowlist_rejection_reason_explains_shell_syntax(self) -> None:
        # A command that matches the pattern but is rejected for shell syntax must
        # say so, not claim it failed to match a pattern.
        provider = LocalCodeExecToolProvider(allowed_commands=[r".*"])

        async with provider:
            syntax_result = await provider.run_command("echo allowed && printf bypassed")
            assignment_result = await provider.run_command("MODE=test echo allowed")

        assert syntax_result.error_kind == "command_not_allowed"
        assert syntax_result.advice is not None
        assert "without a shell" in syntax_result.advice
        assert assignment_result.error_kind == "command_not_allowed"
        assert assignment_result.advice is not None
        assert "assignment" in assignment_result.advice

    @pytest.mark.parametrize(
        ("command", "expected_stdout"),
        [
            ("echo 'a;b && c || d | e > f < g\n$() `x`'", "a;b && c || d | e > f < g\n$() `x`\n"),
            ('echo "it\'s ; literal"', "it's ; literal\n"),
            (r"echo a\;b \$VALUE \${VALUE}", "a;b $VALUE ${VALUE}\n"),
            # Without a shell there is no expansion: variables and globs are
            # passed through as literal argument bytes.
            ('echo "$HOME"', "$HOME\n"),
            ("echo $HOME", "$HOME\n"),
            ("echo *", "*\n"),
            ("echo 'line1\nline2'", "line1\nline2\n"),
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

    async def test_run_command_allowlist_matches_parsed_command_word(self) -> None:
        # Patterns are matched against the parsed command, so quoting inside
        # the command word can neither hide a match nor forge one.
        provider = LocalCodeExecToolProvider(allowed_commands=[r"^echo\b"])

        async with provider:
            quoted = await provider.run_command("ec'ho' allowed")
            forged = await provider.run_command("echo''x allowed")

        assert quoted.error_kind is None
        assert quoted.stdout == "allowed\n"
        assert forged.error_kind == "command_not_allowed"

    async def test_run_command_allowlist_does_not_expand_hostile_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Under Bash, some versions recursively expand array indices, so an
        # attacker-controlled variable could execute code. Without a shell the
        # expansion never happens and the text is echoed back verbatim.
        monkeypatch.setenv("STIRRUP_TEST_INDEX", "$(touch forbidden.txt)")
        provider = LocalCodeExecToolProvider(allowed_commands=[r"^echo"])

        async with provider:
            result = await provider.run_command('echo "${VALUES[STIRRUP_TEST_INDEX]}"')
            side_effect_exists = await provider.file_exists("forbidden.txt")

        assert result.error_kind is None
        assert result.stdout == "${VALUES[STIRRUP_TEST_INDEX]}\n"
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
