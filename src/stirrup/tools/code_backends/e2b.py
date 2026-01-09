"""E2B cloud execution environment backend for code execution."""

from pathlib import Path

from pydantic import ValidationError

try:
    from e2b import InvalidArgumentException, TimeoutException
    from e2b.sandbox.filesystem.filesystem import FileType
    from e2b_code_interpreter import AsyncSandbox, CommandExitException
except ImportError as e:
    raise ImportError(
        "Requires installation of the e2b extra. Install with (for example): `uv pip install stirrup[e2b]` or `uv add stirrup[e2b]`",
    ) from e

import logging

from stirrup.constants import SANDBOX_REQUEST_TIMEOUT, SANDBOX_TIMEOUT
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


class E2BCodeExecToolProvider(CodeExecToolProvider):
    """E2B cloud code execution tool provider.

    Usage with Agent:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        provider = E2BCodeExecToolProvider(timeout=600, template="my-template")
        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(client=client, name="assistant", tools=[provider])
        async with agent.session() as session:
            await session.run("Run Python code")

    Standalone usage:
        provider = E2BCodeExecToolProvider(allowed_commands=[r"^python", r"^pip"])
        async with provider as tool:
            result = await provider.run_command("python script.py")
    """

    def __init__(
        self,
        *,
        timeout: int = SANDBOX_TIMEOUT,
        request_timeout: int = SANDBOX_REQUEST_TIMEOUT,
        template: str | None = None,
        allowed_commands: list[str] | None = None,
    ) -> None:
        """Initialize E2B execution environment configuration.

        Args:
            timeout: Execution environment lifetime in seconds (default: 10 minutes).
            template: Optional E2B template name/alias.
            allowed_commands: Optional list of regex patterns. If provided, only
                             commands matching at least one pattern are allowed.
                             If None, all commands are allowed.

        """
        super().__init__(allowed_commands=allowed_commands)
        self._timeout = timeout
        self._request_timeout = request_timeout
        self._template = template
        self._sbx: AsyncSandbox | None = None

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        """Initialize the E2B sandbox environment and return the code_exec tool."""
        if self._template:
            self._sbx = await AsyncSandbox.create(timeout=self._timeout, template=self._template)
        else:
            self._sbx = await AsyncSandbox.create(timeout=self._timeout)
        return self.get_code_exec_tool()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Cleanup the E2B execution environment."""
        if self._sbx:
            await self._sbx.kill()  # ty: ignore[no-matching-overload]
            self._sbx = None

    async def read_file_bytes(self, path: str) -> bytes:
        """Read file content as bytes from the E2B sandbox.

        Args:
            path: File path within the sandbox.

        Returns:
            File contents as bytes.

        Raises:
            RuntimeError: If environment not started.
            FileNotFoundError: If file does not exist.

        """
        if self._sbx is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        if not await self._sbx.files.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        file_bytes = await self._sbx.files.read(path, format="bytes", request_timeout=self._request_timeout)
        return bytes(file_bytes)

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        """Write bytes to a file in the E2B sandbox.

        Args:
            path: Destination path within the sandbox.
            content: File contents to write.

        Raises:
            RuntimeError: If environment not started.

        """
        if self._sbx is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        await self._sbx.files.write(path, content, request_timeout=self._request_timeout)

    async def run_command(self, cmd: str, *, timeout: int = SHELL_TIMEOUT) -> CommandResult:
        """Execute command in E2B execution environment, returning raw CommandResult."""
        if self._sbx is None:
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

        try:
            r = await self._sbx.commands.run(cmd, timeout=timeout, request_timeout=self._request_timeout)

            return CommandResult(
                exit_code=getattr(r, "exit_code", 0),
                stdout=r.stdout,
                stderr=r.stderr,
            )
        except CommandExitException as exc:
            return CommandResult(
                exit_code=exc.exit_code,
                stdout=exc.stdout,
                stderr=exc.stderr,
            )
        except InvalidArgumentException as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="invalid_argument",
                advice="Avoid NUL/control bytes and nested 'bash -lc'; use a quoted heredoc or base64 write.",
            )
        except TimeoutException as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="timeout",
            )

    async def save_output_files(
        self,
        paths: list[str],
        output_dir: Path | str,
        dest_env: "CodeExecToolProvider | None" = None,
    ) -> SaveOutputFilesResult:
        """Save files from the E2B execution environment to a destination.

        When dest_env is None (local filesystem), files are downloaded from the
        sandbox and saved locally.

        When dest_env is provided (cross-environment transfer), files are copied
        using the base class implementation via read/write primitives.

        Args:
            paths: List of file paths in the execution environment to save.
            output_dir: Directory path to save files to.
            dest_env: If provided, output_dir is interpreted as a path within dest_env
                      (cross-environment transfer). If None, output_dir is a local
                      filesystem path.

        Returns:
            SaveOutputFilesResult containing lists of saved files and any failures.

        """
        if self._sbx is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )

        # If dest_env is provided, use the base class implementation (cross-env transfer)
        if dest_env is not None:
            return await super().save_output_files(paths, output_dir, dest_env)

        # Local filesystem - use optimized E2B API
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        result = SaveOutputFilesResult()

        for env_path in paths:
            try:
                # Check if file exists
                if not await self._sbx.files.exists(env_path):
                    result.failed[env_path] = "File does not exist"
                    logger.warning("Execution environment file does not exist: %s", env_path)
                    continue

                # Get file info to verify it's a file (not a directory)
                info = await self._sbx.files.get_info(env_path)
                if info.type != FileType.FILE:
                    result.failed[env_path] = f"Path is not a file (type: {info.type})"
                    logger.warning("Execution environment path is not a file: %s (type: %s)", env_path, info.type)
                    continue

                # Read file content from execution environment
                file_bytes = await self._sbx.files.read(env_path, format="bytes", request_timeout=self._request_timeout)
                content = bytes(file_bytes)

                # Save with original filename directly in output_dir
                local_path = output_dir_path / Path(env_path).name

                # Write file
                local_path.write_bytes(content)

                result.saved.append(
                    SavedFile(
                        source_path=env_path,
                        output_path=local_path,
                        size=len(content),
                    ),
                )
                logger.debug("Saved file: %s -> %s (%d bytes)", env_path, local_path, len(content))

            except Exception as exc:
                result.failed[env_path] = str(exc)
                logger.exception("Failed to save execution environment file: %s", env_path)

        return result

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: "CodeExecToolProvider | None" = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        """Upload files to the E2B sandbox.

        When source_env is None (local filesystem), files are uploaded via the
        E2B files.write() API.
        Directories are uploaded recursively, preserving their structure.

        When source_env is provided (cross-environment transfer), files are copied
        using the base class implementation via read/write primitives.

        Args:
            *paths: File or directory paths to upload. If source_env is None, these
                    are local filesystem paths. If source_env is provided, these are
                    paths within source_env.
            source_env: If provided, paths are within source_env. If None, paths are
                        local filesystem paths.
            dest_dir: Destination directory in the sandbox.
                      If None, files are placed in /home/user.

        Returns:
            UploadFilesResult containing lists of uploaded files and any failures.

        """
        if self._sbx is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )

        # If source_env is provided, use the base class implementation (cross-env transfer)
        if source_env is not None:
            return await super().upload_files(*paths, source_env=source_env, dest_dir=dest_dir)

        # Local filesystem - use optimized E2B API
        dest_base = dest_dir or "/home/user"
        result = UploadFilesResult()

        for source in paths:
            original_name = Path(source).name  # Get name BEFORE resolve
            source = Path(source).resolve()

            if not source.exists():
                result.failed[str(source)] = "File or directory does not exist"
                logger.warning("Upload source does not exist: %s", source)
                continue

            try:
                if source.is_file():
                    source = Path(source).resolve()
                    dest = f"{dest_base}/{original_name}"
                    content = source.read_bytes()
                    await self._sbx.files.write(dest, content, request_timeout=self._request_timeout)
                    result.uploaded.append(
                        UploadedFile(
                            source_path=source,
                            dest_path=dest,
                            size=len(content),
                        ),
                    )
                    logger.debug("Uploaded file: %s -> %s", source, dest)

                elif source.is_dir():
                    # Upload all files in directory recursively
                    # If dest_dir was explicitly provided, copy contents directly to dest_base
                    # Otherwise, create a subdirectory with the source's name
                    for file_path in source.rglob("*"):
                        if file_path.is_file():
                            relative = file_path.relative_to(source)
                            dest = f"{dest_base}/{relative}" if dest_dir else f"{dest_base}/{source.name}/{relative}"
                            content = file_path.read_bytes()
                            await self._sbx.files.write(dest, content, request_timeout=self._request_timeout)
                            result.uploaded.append(
                                UploadedFile(
                                    source_path=file_path,
                                    dest_path=dest,
                                    size=len(content),
                                ),
                            )
                    if dest_dir:
                        logger.debug("Uploaded directory contents: %s -> %s", source, dest_base)
                    else:
                        logger.debug("Uploaded directory: %s -> %s/%s", source, dest_base, source.name)

            except Exception as exc:
                result.failed[str(source)] = str(exc)
                logger.exception("Failed to upload: %s", source)

        return result

    async def view_image(self, path: str) -> ImageContentBlock:
        """Read and return an image file from the E2B execution environment.

        Args:
            path: Path to image file in the execution environment filesystem.

        Returns:
            ImageContentBlock containing the image data.

        Raises:
            RuntimeError: If execution environment not started.
            FileNotFoundError: If file does not exist.

        """
        if not path.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError(f"Unsupported image type for `{path}`. Only .png, .jpg, or .jpeg are allowed.")

        try:
            file_bytes = await self.read_file_bytes(path)
            image = ImageContentBlock(data=file_bytes)
        except ValidationError as e:
            raise ValueError("You submitted a corrupt/unsupported image file.") from e

        return image
