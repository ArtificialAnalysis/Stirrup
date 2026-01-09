"""Base types and abstract class for code execution backends."""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field

from stirrup.core.models import ImageContentBlock, Tool, ToolProvider, ToolResult, ToolUseCountMetadata
from stirrup.utils.text import truncate_msg

logger = logging.getLogger(__name__)

MAX_LENGTH_SHELL_STDOUT = 20_000
MAX_LENGTH_SHELL_STDERR = 20_000
SHELL_TIMEOUT = 60 * 5


class CodeExecutionParams(BaseModel):
    """Shell command to execute in the execution environment."""

    cmd: Annotated[
        str,
        Field(
            description=(
                "Shell command to execute (bash syntax). "
                "IMPORTANT: Use only relative paths. Do not use absolute paths "
                "(starting with / or ~) or reference directories outside the working directory."
            )
        ),
    ]


@dataclass
class CommandResult:
    """Raw result from command execution (before formatting)."""

    exit_code: int
    stdout: str
    stderr: str
    error_kind: str | None = None  # "invalid_argument", "timeout", etc.
    advice: str | None = None  # Optional advice for error cases


@dataclass
class SavedFile:
    """Information about a file saved from the execution environment."""

    source_path: str  # Original path in execution environment
    output_path: Path  # Path where file was saved
    size: int


@dataclass
class SaveOutputFilesResult:
    """Result of saving output files from the execution environment."""

    saved: list[SavedFile] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # source_path -> error message


@dataclass
class UploadedFile:
    """Information about a file uploaded to the execution environment."""

    source_path: Path  # Original path on local filesystem
    dest_path: str  # Path in the execution environment
    size: int


class ViewImageParams(BaseModel):
    """Parameters for viewing an image from the execution environment."""

    path: Annotated[
        str,
        Field(
            description="Path to the image file within the execution environment filesystem (supports .png, .jpg, .jpeg, .gif, .bmp, .tiff, .psd)"
        ),
    ]


@dataclass
class UploadFilesResult:
    """Result of uploading files to the execution environment."""

    uploaded: list[UploadedFile] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # source_path -> error message


def format_result(result: CommandResult) -> ToolResult[ToolUseCountMetadata]:
    """Format a CommandResult as XML ToolResult (shared by all backends)."""
    if result.error_kind:
        # Error case
        content = (
            f"<shell_results>"
            f"\n<error_kind>{result.error_kind}</error_kind>"
            f"\n<details>{truncate_msg(result.stderr, MAX_LENGTH_SHELL_STDERR)}</details>"
        )
        if result.advice:
            content += f"\n<advice>{result.advice}</advice>"
        content += "\n</shell_results>"
    else:
        # Success case
        content = (
            f"<shell_results>"
            f"\n<exit_code>{result.exit_code}</exit_code>"
            f"\n<stdout>{truncate_msg(result.stdout, MAX_LENGTH_SHELL_STDOUT)}</stdout>"
            f"\n<stderr>{truncate_msg(result.stderr, MAX_LENGTH_SHELL_STDERR)}</stderr>"
            f"\n</shell_results>"
        )
    return ToolResult(content=content, metadata=ToolUseCountMetadata())


class CodeExecToolProvider(ToolProvider, ABC):
    """Abstract base class for code execution tool providers.

    CodeExecToolProvider is a ToolProvider that manages code execution environments
    (sandboxes, containers, local temp directories) and returns a code_exec Tool.

    Subclasses must implement:
    - __aenter__(): Initialize environment and return the code_exec tool
    - __aexit__(): Cleanup the execution environment
    - run_command(): Execute a command and return raw result
    - read_file_bytes(): Read file content as bytes from the environment
    - write_file_bytes(): Write bytes to a file in the environment

    Default implementations are provided for:
    - save_output_files(): Save files to local dir or another exec env (uses primitives)
    - upload_files(): Upload files from local or another exec env (uses primitives)

    All code execution providers support an optional allowlist of command patterns.
    If provided, only commands matching at least one pattern are allowed.
    If None, all commands are allowed.

    Usage with Agent:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(
            client=client,
            name="assistant",
            tools=[LocalCodeExecToolProvider(), CALCULATOR_TOOL],
        )
    """

    def __init__(self, *, allowed_commands: list[str] | None = None) -> None:
        """Initialize execution environment with optional command allowlist.

        Args:
            allowed_commands: Optional list of regex patterns. If provided, only
                             commands matching at least one pattern are allowed.
                             If None, all commands are allowed.

        """
        self._allowed_commands = allowed_commands
        self._compiled_allowed: list[re.Pattern[str]] | None = None
        if allowed_commands is not None:
            self._compiled_allowed = [re.compile(p) for p in allowed_commands]

    @property
    def temp_dir(self) -> Path | None:
        """Return the temporary directory for this execution environment, if any."""
        return None

    def _check_allowed(self, cmd: str) -> bool:
        """Check if command is allowed based on the allowlist.

        Returns:
            True if the command is allowed, False otherwise.

        """
        if self._compiled_allowed is None:
            return True  # No allowlist = allow all
        return any(p.search(cmd) for p in self._compiled_allowed)

    @abstractmethod
    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        """Enter async context: set up environment and return code_exec tool."""
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context: cleanup the execution environment."""
        ...

    @abstractmethod
    async def run_command(self, cmd: str, *, timeout: int = SHELL_TIMEOUT) -> CommandResult:
        """Execute a shell command and return raw CommandResult."""
        ...

    @abstractmethod
    async def read_file_bytes(self, path: str) -> bytes:
        """Read file content as bytes from this execution environment.

        Args:
            path: File path within this execution environment (relative or absolute
                  within the env's working directory).

        Returns:
            File contents as bytes.

        Raises:
            FileNotFoundError: If file does not exist.
            RuntimeError: If execution environment not started.

        """
        ...

    @abstractmethod
    async def write_file_bytes(self, path: str, content: bytes) -> None:
        """Write bytes to a file in this execution environment.

        Args:
            path: Destination path within this execution environment.
            content: File contents to write.

        Raises:
            RuntimeError: If execution environment not started.

        """
        ...

    async def save_output_files(
        self,
        paths: list[str],
        output_dir: Path | str,
        dest_env: "CodeExecToolProvider | None" = None,
    ) -> SaveOutputFilesResult:
        """Save files from this execution environment to a destination.

        Args:
            paths: List of file paths in this execution environment to save.
            output_dir: Directory path to save files to.
            dest_env: If provided, output_dir is interpreted as a path within dest_env
                      (cross-environment transfer). If None, output_dir is a local
                      filesystem path.

        Returns:
            SaveOutputFilesResult containing lists of saved files and any failures.

        """
        result = SaveOutputFilesResult()
        output_dir_str = str(output_dir)

        for source_path in paths:
            try:
                content = await self.read_file_bytes(source_path)
                filename = Path(source_path).name
                dest_path = f"{output_dir_str}/{filename}"

                if dest_env:
                    # Transfer to another exec env (cross-environment)
                    logger.debug(
                        "CROSS-ENV TRANSFER: %s (%d bytes) -> %s (dest_env: %s)",
                        source_path,
                        len(content),
                        dest_path,
                        type(dest_env).__name__,
                    )
                    await dest_env.write_file_bytes(dest_path, content)
                    result.saved.append(SavedFile(source_path, Path(dest_path), len(content)))
                else:
                    # Save to local filesystem
                    output_path = Path(output_dir) / filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    logger.debug(
                        "SAVE TO LOCAL: %s (%d bytes) -> %s",
                        source_path,
                        len(content),
                        output_path,
                    )
                    output_path.write_bytes(content)
                    result.saved.append(SavedFile(source_path, output_path, len(content)))
            except Exception as e:
                logger.debug("TRANSFER FAILED: %s -> %s: %s", source_path, output_dir_str, e)
                result.failed[source_path] = str(e)

        return result

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: "CodeExecToolProvider | None" = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        """Upload files to this execution environment.

        Args:
            *paths: File or directory paths to upload. If source_env is None, these
                    are local filesystem paths. If source_env is provided, these are
                    paths within source_env (cross-environment transfer).
            source_env: If provided, paths are within source_env. If None, paths are
                        local filesystem paths.
            dest_dir: Destination directory in this environment.
                      If None, uses the environment's working directory.

        Returns:
            UploadFilesResult containing lists of uploaded files and any failures.

        Raises:
            RuntimeError: If execution environment not started.

        """
        result = UploadFilesResult()
        dest_dir_str = dest_dir or ""

        for path in paths:
            path_str = str(path)
            try:
                if source_env:
                    # Cross-environment transfer: read from source_env
                    content = await source_env.read_file_bytes(path_str)
                    filename = Path(path_str).name
                    dest_path = f"{dest_dir_str}/{filename}" if dest_dir_str else filename
                    logger.debug(
                        "UPLOAD CROSS-ENV: %s (%d bytes) from %s -> %s",
                        path_str,
                        len(content),
                        type(source_env).__name__,
                        dest_path,
                    )
                    await self.write_file_bytes(dest_path, content)
                    result.uploaded.append(UploadedFile(Path(path_str), dest_path, len(content)))
                else:
                    # Local filesystem upload - must be handled by subclass
                    # This is a fallback that reads from local fs and writes to env
                    local_path = Path(path)
                    if local_path.is_dir():
                        # Handle directory recursively
                        for file_path in local_path.rglob("*"):
                            if file_path.is_file():
                                rel_path = file_path.relative_to(local_path)
                                dest_path = f"{dest_dir_str}/{rel_path}" if dest_dir_str else str(rel_path)
                                content = file_path.read_bytes()
                                logger.debug(
                                    "UPLOAD FROM LOCAL: %s (%d bytes) -> %s",
                                    file_path,
                                    len(content),
                                    dest_path,
                                )
                                await self.write_file_bytes(dest_path, content)
                                result.uploaded.append(UploadedFile(file_path, dest_path, len(content)))
                    else:
                        filename = local_path.name
                        dest_path = f"{dest_dir_str}/{filename}" if dest_dir_str else filename
                        content = local_path.read_bytes()
                        logger.debug(
                            "UPLOAD FROM LOCAL: %s (%d bytes) -> %s",
                            local_path,
                            len(content),
                            dest_path,
                        )
                        await self.write_file_bytes(dest_path, content)
                        result.uploaded.append(UploadedFile(local_path, dest_path, len(content)))
            except Exception as e:
                logger.debug("UPLOAD FAILED: %s -> %s: %s", path_str, dest_dir_str, e)
                result.failed[path_str] = str(e)

        return result

    def get_code_exec_tool(
        self,
        *,
        name: str = "code_exec",
        description: str | None = None,
    ) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        """Create a code execution tool for this environment.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            Tool[CodeExecutionParams] that executes commands in this environment

        """
        env = self

        async def executor(params: CodeExecutionParams) -> ToolResult[ToolUseCountMetadata]:
            result = await env.run_command(params.cmd)
            return format_result(result)

        return Tool[CodeExecutionParams, ToolUseCountMetadata](
            name=name,
            description=description
            or "Execute a shell command in the execution environment. Returns exit code, stdout, and stderr as XML.",
            parameters=CodeExecutionParams,
            executor=executor,  # ty: ignore[invalid-argument-type]
        )

    def get_view_image_tool(
        self,
        *,
        name: str = "view_image",
        description: str | None = None,
    ) -> Tool[ViewImageParams, ToolUseCountMetadata]:
        """Create a view_image tool for this environment.

        Args:
            name: Tool name
            description: Tool description

        Returns:
            Tool[ViewImageParams, ToolUseCountMetadata] that views images in this environment

        """
        env = self

        async def executor(params: ViewImageParams) -> ToolResult[ToolUseCountMetadata]:
            try:
                image = await env.view_image(params.path)
                return ToolResult(
                    content=["Viewing image at path: " + params.path, image],
                    metadata=ToolUseCountMetadata(),
                )
            except FileNotFoundError:
                return ToolResult(
                    content=f"Image `{params.path}` not found.",
                    success=False,
                    metadata=ToolUseCountMetadata(),
                )
            except ValueError as e:
                return ToolResult(
                    content=str(e),
                    success=False,
                    metadata=ToolUseCountMetadata(),
                )

        return Tool[ViewImageParams, ToolUseCountMetadata](
            name=name,
            description=description or "View an image file from the execution environment's filesystem.",
            parameters=ViewImageParams,
            executor=executor,  # ty: ignore[invalid-argument-type]
        )

    @abstractmethod
    async def view_image(self, path: str) -> ImageContentBlock:
        """Read and return an image file from the execution environment.

        Args:
            path: Path to image file in the execution environment (relative or absolute).

        Returns:
            ImageContentBlock containing the image data.

        Raises:
            RuntimeError: If execution environment not started.
            FileNotFoundError: If file does not exist.
            ValueError: If path is outside the execution environment, is a directory,
                        or the file is not a valid image.

        """
        ...
