"""Local execution environment backend for code execution in an isolated temp directory."""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import anyio

from stirrup.core.models import ImageContentBlock, Tool, ToolUseCountMetadata

from .base import (
    SHELL_TIMEOUT,
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
    SavedFile,
    SaveOutputFilesResult,
    UploadedFile,
    UploadFilesResult,
)

logger = logging.getLogger(__name__)


class LocalCodeExecToolProvider(CodeExecToolProvider):
    """Local code execution tool provider using an isolated temp directory.

    Commands are executed with the temp directory as the working directory.
    An optional allowlist can restrict which commands are permitted.

    Usage with Agent:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(
            client=client,
            name="assistant",
            tools=[LocalCodeExecToolProvider(), CALCULATOR_TOOL],
        )

        async with agent.session(output_dir="./output") as session:
            await session.run("Run some Python code")

    Standalone usage:
        provider = LocalCodeExecToolProvider()

        async with provider as tool:
            # tool is a Tool instance for code execution
            result = await provider.run_command("python script.py")
            await provider.save_output_files(["output.txt"], "/path/to/output")
    """

    def __init__(
        self,
        *,
        allowed_commands: list[str] | None = None,
        temp_base_dir: Path | str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize LocalCodeExecToolProvider configuration.

        Args:
            allowed_commands: Optional list of regex patterns. If provided, only
                             commands matching at least one pattern are allowed.
                             If None, all commands are allowed.
            temp_base_dir: Optional base directory for creating the execution environment
                          temp directory. If None, uses the system default temp directory.
            description: Optional description of the tool. If None, uses the default description.

        """
        super().__init__(allowed_commands=allowed_commands)
        self._temp_dir: Path | None = None
        self._temp_base_dir: Path | None = Path(temp_base_dir) if temp_base_dir else None
        self._description = (
            description
            or "Execute a shell command in the execution environment. Returns exit code, stdout, and stderr as XML. Use `uv` to manage packages."
        )

    @property
    def temp_dir(self) -> Path | None:
        """Return the temp directory path, or None if not started."""
        return self._temp_dir

    def _check_absolute_paths(self, cmd: str) -> CommandResult | None:
        """Check if command contains absolute paths that could escape the temp directory.

        Returns:
            CommandResult with error if absolute paths detected, None otherwise.

        Note:
            This check is specific to LocalCodeExecToolProvider since Docker and E2B
            providers are already sandboxed and absolute paths are safe within them.
        """
        absolute_patterns = [
            r"~/",  # ~/path - home directory shortcut
            r"/(?:home|Users|tmp|var|etc)/",  # /home/, /Users/, /tmp/, etc.
            r"\$HOME/",  # $HOME/path
            r"\$\{HOME\}/",  # ${HOME}/path
        ]
        for pattern in absolute_patterns:
            if re.search(pattern, cmd):
                return CommandResult(
                    exit_code=1,
                    stdout="",
                    stderr=(
                        "Command appears to use absolute paths which could write outside "
                        "the execution environment. Use relative paths instead."
                    ),
                    error_kind="absolute_path_detected",
                    advice=(
                        "Use relative paths (e.g., './output.txt' instead of '~/output.txt'). "
                        "For full filesystem access, use DockerCodeExecToolProvider or E2BCodeExecToolProvider."
                    ),
                )
        return None

    async def __aenter__(self) -> "Tool[CodeExecutionParams, ToolUseCountMetadata]":
        """Create temp directory and return the code_exec tool."""
        if self._temp_base_dir:
            self._temp_base_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = Path(tempfile.mkdtemp(prefix="local_exec_env_", dir=self._temp_base_dir))
        logger.debug("Created local execution environment temp directory: %s", self._temp_dir)
        return self.get_code_exec_tool(description=self._description)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Cleanup the local execution environment."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as exc:
                logger.warning("Failed to cleanup temp directory %s: %s", self._temp_dir, exc)
        self._temp_dir = None

    def _resolve_and_validate_path(self, path: str) -> Path:
        """Resolve a path and validate it's within the temp directory.

        Args:
            path: File path (relative or absolute within the temp dir).

        Returns:
            Resolved absolute Path.

        Raises:
            RuntimeError: If environment not started.
            ValueError: If path is outside temp directory.
            FileNotFoundError: If path does not exist (for reads).

        """
        if self._temp_dir is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = self._temp_dir / resolved

        # Security: ensure path is within temp directory
        try:
            resolved.resolve().relative_to(self._temp_dir.resolve())
        except ValueError as e:
            raise ValueError(f"Path is outside execution environment: {path}") from e

        return resolved

    async def read_file_bytes(self, path: str) -> bytes:
        """Read file content as bytes from the temp directory.

        Args:
            path: File path (relative or absolute within the temp dir).

        Returns:
            File contents as bytes.

        Raises:
            RuntimeError: If environment not started.
            ValueError: If path is outside temp directory.
            FileNotFoundError: If file does not exist.

        """
        resolved = self._resolve_and_validate_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return resolved.read_bytes()

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        """Write bytes to a file in the temp directory.

        Args:
            path: Destination path (relative or absolute within the temp dir).
            content: File contents to write.

        Raises:
            RuntimeError: If environment not started.
            ValueError: If path is outside temp directory.

        """
        resolved = self._resolve_and_validate_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_bytes(content)

    async def run_command(self, cmd: str, *, timeout: int = SHELL_TIMEOUT) -> CommandResult:
        """Execute command in the temp directory.

        Args:
            cmd: Shell command to execute (bash syntax).
            timeout: Maximum time in seconds to wait for command completion.

        Returns:
            CommandResult with exit_code, stdout, stderr, and optional error info.

        """
        if self._temp_dir is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )

        # Check allowlist
        if not self._check_allowed(cmd):
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: '{cmd}' does not match any allowed patterns",
                error_kind="command_not_allowed",
                advice="Only commands matching the allowlist patterns are permitted.",
            )

        # Check for absolute paths (local environment is not sandboxed)
        absolute_path_error = self._check_absolute_paths(cmd)
        if absolute_path_error:
            return absolute_path_error

        process = None
        try:
            with anyio.fail_after(timeout):
                # Use shell=True by wrapping in a shell command
                process = await anyio.open_process(
                    ["bash", "-c", cmd],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=self._temp_dir,
                )

                # Read all output from streams concurrently
                stdout_chunks: list[bytes] = []
                stderr_chunks: list[bytes] = []

                async def read_stdout() -> None:
                    if process.stdout:
                        stdout_chunks.extend([chunk async for chunk in process.stdout])

                async def read_stderr() -> None:
                    if process.stderr:
                        stderr_chunks.extend([chunk async for chunk in process.stderr])

                async with anyio.create_task_group() as tg:
                    tg.start_soon(read_stdout)
                    tg.start_soon(read_stderr)

                await process.wait()

                return CommandResult(
                    exit_code=process.returncode or 0,
                    stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
                    stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
                )

        except TimeoutError:
            if process:
                process.kill()
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                error_kind="timeout",
            )
        except Exception as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="execution_error",
            )

    async def save_output_files(
        self,
        paths: list[str],
        output_dir: Path | str,
        dest_env: "CodeExecToolProvider | None" = None,
    ) -> SaveOutputFilesResult:
        """Move files from the temp directory to a destination.

        When dest_env is None (local filesystem), files are MOVED (not copied) -
        originals are deleted from the execution environment.
        Existing files in output_dir are silently overwritten.

        When dest_env is provided (cross-environment transfer), files are copied
        using the base class implementation via read/write primitives.

        Args:
            paths: List of file paths in the execution environment (relative or absolute).
                   Relative paths are resolved against the execution environment temp directory.
            output_dir: Directory path to save files to.
            dest_env: If provided, output_dir is interpreted as a path within dest_env
                      (cross-environment transfer). If None, output_dir is a local
                      filesystem path.

        Returns:
            SaveOutputFilesResult containing lists of saved files and any failures.

        """
        if self._temp_dir is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )

        # If dest_env is provided, use the base class implementation (cross-env transfer)
        if dest_env is not None:
            return await super().save_output_files(paths, output_dir, dest_env)

        # Local filesystem - use optimized move operation
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        result = SaveOutputFilesResult()

        for source_path_str in paths:
            try:
                source_path = Path(source_path_str)
                if not source_path.is_absolute():
                    source_path = self._temp_dir / source_path

                # Security: ensure path is within temp directory
                try:
                    source_path.resolve().relative_to(self._temp_dir.resolve())
                except ValueError:
                    result.failed[source_path_str] = "Path is outside execution environment directory"
                    logger.warning("Attempted to access path outside execution environment: %s", source_path_str)
                    continue

                if not source_path.exists():
                    result.failed[source_path_str] = "File does not exist"
                    logger.warning("Execution environment file does not exist: %s", source_path_str)
                    continue

                if not source_path.is_file():
                    result.failed[source_path_str] = "Path is not a file"
                    logger.warning("Execution environment path is not a file: %s", source_path_str)
                    continue

                file_size = source_path.stat().st_size
                dest_path = output_dir_path / source_path.name

                # Move file (overwrites if exists)
                shutil.move(str(source_path), str(dest_path))
                logger.debug("Moved file: %s -> %s", source_path, dest_path)

                result.saved.append(
                    SavedFile(
                        source_path=source_path_str,
                        output_path=dest_path,
                        size=file_size,
                    ),
                )

            except Exception as exc:
                result.failed[source_path_str] = str(exc)
                logger.exception("Failed to move file: %s", source_path_str)

        return result

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: "CodeExecToolProvider | None" = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        """Upload files to the execution environment.

        When source_env is None (local filesystem), files are COPIED (not moved) -
        originals remain on the local filesystem.
        Directories are uploaded recursively, preserving their structure.

        When source_env is provided (cross-environment transfer), files are copied
        using the base class implementation via read/write primitives.

        Args:
            *paths: File or directory paths to upload. If source_env is None, these
                    are local filesystem paths. If source_env is provided, these are
                    paths within source_env.
            source_env: If provided, paths are within source_env. If None, paths are
                        local filesystem paths.
            dest_dir: Destination subdirectory within the temp directory.
                      If None, files are placed directly in the temp directory.

        Returns:
            UploadFilesResult containing lists of uploaded files and any failures.

        """
        if self._temp_dir is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )

        # If source_env is provided, use the base class implementation (cross-env transfer)
        if source_env is not None:
            return await super().upload_files(*paths, source_env=source_env, dest_dir=dest_dir)

        # Local filesystem - use optimized copy operation
        dest_base = self._temp_dir / dest_dir if dest_dir else self._temp_dir
        dest_base.mkdir(parents=True, exist_ok=True)

        result = UploadFilesResult()

        for source in paths:
            source = Path(source).resolve()

            if not source.exists():
                result.failed[str(source)] = "File or directory does not exist"
                logger.warning("Upload source does not exist: %s", source)
                continue

            try:
                if source.is_file():
                    dest = dest_base / source.name
                    shutil.copy2(source, dest)
                    result.uploaded.append(
                        UploadedFile(
                            source_path=source,
                            dest_path=str(dest.relative_to(self._temp_dir)),
                            size=source.stat().st_size,
                        ),
                    )
                    logger.debug("Uploaded file: %s -> %s", source, dest)

                elif source.is_dir():
                    # If dest_dir was explicitly provided, copy contents directly to dest_base
                    # Otherwise, create a subdirectory with the source's name
                    if dest_dir:
                        dest = dest_base
                        # Copy contents of source directory into dest_base
                        for item in source.iterdir():
                            item_dest = dest / item.name
                            if item.is_file():
                                shutil.copy2(item, item_dest)
                            else:
                                shutil.copytree(item, item_dest, dirs_exist_ok=True)
                    else:
                        dest = dest_base / source.name
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                    # Track all individual files uploaded
                    for file_path in source.rglob("*"):
                        if file_path.is_file():
                            relative = file_path.relative_to(source)
                            dest_file = dest / relative
                            result.uploaded.append(
                                UploadedFile(
                                    source_path=file_path,
                                    dest_path=str(dest_file.relative_to(self._temp_dir)),
                                    size=file_path.stat().st_size,
                                ),
                            )
                    logger.debug("Uploaded directory: %s -> %s", source, dest)

            except Exception as exc:
                result.failed[str(source)] = str(exc)
                logger.exception("Failed to upload: %s", source)

        return result

    async def view_image(self, path: str) -> ImageContentBlock:
        """Read and return an image file from the local execution environment.

        Args:
            path: Path to image file (relative to temp directory, or absolute within it).

        Returns:
            ImageContentBlock containing the image data.

        Raises:
            RuntimeError: If execution environment not started.
            FileNotFoundError: If file does not exist.
            ValueError: If path is outside temp directory, is a directory, or not a valid image.

        """
        file_bytes = await self.read_file_bytes(path)
        return ImageContentBlock(data=file_bytes)
