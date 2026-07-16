"""Base types and abstract class for code execution backends."""

import logging
import os
import re
import shlex
import shutil
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePath, PurePosixPath
from stat import S_IMODE
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from stirrup.core.models import ImageContentBlock, Tool, ToolProvider, ToolResult, ToolUseCountMetadata
from stirrup.utils.text import truncate_msg

logger = logging.getLogger(__name__)

MAX_LENGTH_SHELL_STDOUT = 20_000
MAX_LENGTH_SHELL_STDERR = 20_000
SHELL_TIMEOUT = 60 * 5

# Reason a command is rejected by the allowlist; each key maps to advice below.
RejectionReason = Literal["empty_command", "no_pattern_match", "shell_syntax", "shell_assignment"]

# Shell operator characters: separators, pipes, redirects, subshell parens,
# backticks, and newline. Appearing unquoted, they mean the command expects a
# shell — but allowlisted commands run without one, so we reject rather than
# pass the operator through as a do-nothing literal argument.
_SHELL_PUNCTUATION = ";<>|&()`\n"

# Advice shown to the model for each rejection reason.
_COMMAND_NOT_ALLOWED_ADVICE: dict[RejectionReason, str] = {
    "empty_command": "The command is empty.",
    "no_pattern_match": (
        "The command does not match any allowlist pattern. Patterns are matched "
        "against the parsed command, so quoting does not affect matching."
    ),
    "shell_syntax": (
        "Allowlisted commands run without a shell. Shell syntax such as ;, &&, |, "
        "redirects, $(...), backticks, or multiple lines is not supported. Run one "
        "command with literal arguments."
    ),
    "shell_assignment": (
        "Assignment prefixes such as NAME=value are not supported. Run the command without the leading assignment."
    ),
}


@dataclass(frozen=True)
class OutputSourceRoot:
    """An absolute execution-environment root accepted when resolving output paths.

    Attributes:
        path: Absolute POSIX root path (execution environments are Linux).
        host_path: Set when this root is a directory on the host filesystem.
            Enables canonical (symlink-aware) resolution, e.g. accepting
            ``/tmp/...`` spellings of a ``/private/tmp/...`` root on macOS.
    """

    path: str
    host_path: Path | None = None

    @classmethod
    def for_host(cls, path: Path) -> "OutputSourceRoot":
        """Build a root backed by the host filesystem."""
        return cls(path=str(path), host_path=path)


def _safe_output_relative_path(
    source_path: str,
    *,
    source_roots: tuple[OutputSourceRoot, ...] = (),
) -> PurePosixPath:
    """Return a safe POSIX path relative to an output destination.

    Relative paths retain their directory structure. Absolute paths are made
    relative to the first matching backend root; for host-backed roots a
    symlink-equivalent (canonical) spelling of the root is also accepted.

    Raises:
        ValueError: If the path is empty, contains NUL or traversal, does not
            identify a file, or is absolute but outside every declared root.
    """
    if not source_path or "\x00" in source_path:
        raise ValueError("Output source path is empty or invalid")

    source = PurePosixPath(source_path)
    if ".." in source.parts:
        raise ValueError(f"Output source path contains traversal: {source_path}")
    if not source.name:
        raise ValueError(f"Output source path does not identify a file: {source_path}")

    relative = _relative_to_canonical_host_root(source_path, source_roots)
    if relative is None:
        relative = source
        if source.is_absolute():
            for root in (PurePosixPath(item.path) for item in source_roots):
                try:
                    relative = source.relative_to(root)
                except ValueError:
                    continue
                break
            else:
                raise ValueError(f"Output source path is outside known execution roots: {source_path}")

    if not relative.name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Output source path cannot be safely preserved: {source_path}")
    return relative


def _relative_to_canonical_host_root(
    source_path: str,
    source_roots: tuple[OutputSourceRoot, ...],
) -> PurePosixPath | None:
    """Preserve a lexical suffix beneath an equivalent canonical host root."""
    source = Path(source_path)
    if not source.is_absolute():
        return None

    for root in source_roots:
        if root.host_path is None:
            continue
        canonical_root = root.host_path.resolve(strict=False)
        for ancestor in (source, *source.parents):
            if ancestor.resolve(strict=False) == canonical_root:
                return PurePosixPath(*source.relative_to(ancestor).parts)
    return None


def _is_case_sensitive_filesystem(directory: Path) -> bool:
    """Probe whether names differing only by case are distinct in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, probe_name = tempfile.mkstemp(prefix=".stirrup-Case-Probe-", dir=directory)
    os.close(descriptor)
    probe = Path(probe_name)
    alternate = probe.with_name(probe.name.swapcase())
    try:
        return not alternate.exists() or not os.path.samefile(probe, alternate)
    finally:
        probe.unlink(missing_ok=True)


def _host_output_destination_identity(destination: Path, output_root: Path) -> str:
    """Validate a host destination and return its filesystem-aware identity."""
    resolved_root = output_root.resolve(strict=False)
    resolved_destination = destination.resolve(strict=False)
    try:
        resolved_destination.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Output destination escapes output directory: {destination}") from exc

    probe_directory = resolved_destination.parent
    while not probe_directory.exists() and probe_directory != resolved_root:
        probe_directory = probe_directory.parent

    identity = str(resolved_destination)
    return identity if _is_case_sensitive_filesystem(probe_directory) else identity.casefold()


def _reserve_output_destination(
    destination: Path,
    reserved: dict[str, str],
    *,
    claimant: str,
    output_root: Path,
) -> None:
    """Reserve one canonical destination path for ``claimant``."""
    identity = _host_output_destination_identity(destination, output_root)
    if identity in reserved:
        raise ValueError(f"Output destination collision: {destination} (already claimed by {reserved[identity]})")
    reserved[identity] = claimant


def _unquoted_operator_tokens(cmd: str) -> list[str]:
    """Return any unquoted shell operator tokens in cmd.

    Uses shlex punctuation mode, which splits runs of operator characters into
    their own tokens unless they are quoted or escaped. This catches operators
    attached to a word (``>out.txt``, ``foo;rm``) while leaving quoted literals
    (``'a;b'``, ``--format='%h;%s'``) alone. Newline counts as an operator, not
    whitespace. Raises ValueError on unbalanced quotes.
    """
    lex = shlex.shlex(cmd, posix=True, punctuation_chars=_SHELL_PUNCTUATION)
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    return [token for token in lex if token and all(c in _SHELL_PUNCTUATION for c in token)]


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
    output_path: PurePath  # Concrete locally; PurePosixPath in another execution environment
    size: int


@dataclass(frozen=True)
class _CapturedOutputFile:
    """One host-backed output captured before destination writes begin."""

    source_path: str
    output_path: Path
    snapshot_path: Path
    size: int
    mode: int


@dataclass
class SaveOutputFilesResult:
    """Result of saving output files from the execution environment."""

    saved: list[SavedFile] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # source_path -> error message


def _atomic_copy(snapshot: Path, destination: Path, mode: int) -> None:
    """Copy a snapshot into place without exposing a partial destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The temp name must stay valid even when the destination name is already
    # at the filesystem's maximum length.
    descriptor, temporary_name = tempfile.mkstemp(prefix=".stirrup-out-", dir=destination.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(snapshot, temporary_path)
        temporary_path.chmod(mode)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_bytes(content: bytes, destination: Path) -> None:
    """Write bytes atomically to a local output destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The temp name must stay valid even when the destination name is already
    # at the filesystem's maximum length.
    descriptor, temporary_name = tempfile.mkstemp(prefix=".stirrup-out-", dir=destination.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _save_host_output_files(
    paths: list[str],
    output_dir: Path | str,
    *,
    source_roots: tuple[OutputSourceRoot, ...],
    resolve_source: Callable[[str], Path],
) -> SaveOutputFilesResult:
    """Copy host-backed outputs while preserving safe source-relative paths.

    Sources are captured before any destination is written so an output
    directory may overlap the execution directory without request ordering
    changing the copied bytes. Sources remain in the execution environment.
    """
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    result = SaveOutputFilesResult()
    planned_destinations: dict[str, str] = {}
    captured_files: list[_CapturedOutputFile] = []

    with tempfile.TemporaryDirectory(prefix="stirrup-output-snapshot-") as snapshot_dir_str:
        snapshot_dir = Path(snapshot_dir_str)
        for index, source_path in enumerate(dict.fromkeys(paths)):
            try:
                relative_path = _safe_output_relative_path(source_path, source_roots=source_roots)
                source_fs_path = resolve_source(source_path)
                if not source_fs_path.exists():
                    raise FileNotFoundError("File does not exist")
                if not source_fs_path.is_file():
                    raise ValueError("Path is not a file")

                destination = output_root.joinpath(*relative_path.parts)
                _reserve_output_destination(
                    destination,
                    planned_destinations,
                    claimant=source_path,
                    output_root=output_root,
                )
                source_stat = source_fs_path.stat()
                snapshot_path = snapshot_dir / str(index)
                shutil.copyfile(source_fs_path, snapshot_path)
                captured_files.append(
                    _CapturedOutputFile(
                        source_path=source_path,
                        output_path=destination,
                        snapshot_path=snapshot_path,
                        size=snapshot_path.stat().st_size,
                        mode=S_IMODE(source_stat.st_mode),
                    )
                )
            except (ValueError, FileNotFoundError) as exc:
                result.failed[source_path] = str(exc)
                logger.warning("Rejected output file %s: %s", source_path, exc)
            except Exception as exc:
                result.failed[source_path] = str(exc)
                logger.exception("Failed to capture output file: %s", source_path)

        for captured in captured_files:
            try:
                _atomic_copy(captured.snapshot_path, captured.output_path, captured.mode)
                result.saved.append(SavedFile(captured.source_path, captured.output_path, captured.size))
            except Exception as exc:
                result.failed[captured.source_path] = str(exc)
                logger.exception("Failed to save captured file: %s", captured.source_path)

    return result


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

    Overridable output-path hooks (see the extending docs for the contract):
    - output_source_roots(): Declare absolute roots for resolving output paths
    - resolve_output_source(): Resolve/reject a source before it is read
    - output_destination_identity(): Reject escapes and identify cross-env destinations

    All providers accept an optional allowlist of command patterns. With an
    allowlist, each command is parsed with shlex, matched against the patterns,
    and executed without a shell: arguments are passed literally, and shell
    syntax such as pipes, redirects, and expansions is rejected. Without an
    allowlist, all commands are allowed and run through a shell.

    Usage with Agent:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(
            client=client,
            name="assistant",
            tools=[LocalCodeExecToolProvider(), CALCULATOR_TOOL],
        )
    """

    def __init__(
        self,
        *,
        allowed_commands: list[str] | None = None,
        shell_timeout: int = SHELL_TIMEOUT,
    ) -> None:
        """Initialize execution environment with optional command allowlist.

        Args:
            allowed_commands: Optional list of regex patterns, matched from the
                             start of the parsed command. When set, commands run
                             without a shell: arguments are literal and shell
                             syntax is rejected (see ``_prepare_command``). When
                             None, all commands are allowed and run through a
                             shell.
            shell_timeout: Per-command wall-clock timeout (seconds) applied to
                           every ``code_exec`` invocation from the LLM. Defaults
                           to ``SHELL_TIMEOUT``. Callers should set this to match
                           their application's expected long-running-command
                           budget rather than rely on the stirrup default.

        """
        self._allowed_commands = allowed_commands
        self._shell_timeout = shell_timeout
        self._compiled_allowed: list[re.Pattern[str]] | None = None
        if allowed_commands is not None:
            self._compiled_allowed = [re.compile(p) for p in allowed_commands]

    @property
    def temp_dir(self) -> Path | None:
        """Return the temporary directory for this execution environment, if any."""
        return None

    def _prepare_command(self, cmd: str) -> tuple[list[str] | None, CommandResult | None]:
        """Check cmd against the allowlist and parse it for execution.

        Returns ``(argv, rejection)``, where at most one side is set:

        - ``(None, None)``: no allowlist configured. Run ``cmd`` through a
          shell as usual.
        - ``(None, rejection)``: the command was rejected. Return the rejection.
        - ``(argv, None)``: the command is allowed. Execute ``argv`` directly,
          with no shell involved.

        Patterns are matched against ``shlex.join(argv)`` — the canonical form
        of what will actually run — so quoting tricks cannot make a pattern see
        a different command than the one executed.
        """
        if self._compiled_allowed is None:
            return None, None
        try:
            argv = shlex.split(cmd)
            operator_tokens = _unquoted_operator_tokens(cmd)
        except ValueError:
            return None, self._rejection(cmd, "shell_syntax")
        if not argv:
            return None, self._rejection(cmd, "empty_command")
        if operator_tokens:
            return None, self._rejection(cmd, "shell_syntax")
        if "=" in argv[0]:
            return None, self._rejection(cmd, "shell_assignment")
        canonical = shlex.join(argv)
        if not any(p.match(canonical) for p in self._compiled_allowed):
            return None, self._rejection(cmd, "no_pattern_match")
        return argv, None

    def _rejection(self, cmd: str, reason: RejectionReason) -> CommandResult:
        """Build the CommandResult for an allowlist rejection."""
        logger.debug("command rejected (%s): %r", reason, cmd)
        return CommandResult(
            exit_code=1,
            stdout="",
            stderr=f"Command not allowed: '{cmd}'",
            error_kind="command_not_allowed",
            advice=_COMMAND_NOT_ALLOWED_ADVICE[reason],
        )

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
    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        """Execute a shell command and return raw CommandResult.

        Args:
            cmd: Shell command to execute (bash syntax).
            timeout: Per-call wall-clock timeout (seconds). If None, falls back
                to ``self._shell_timeout`` configured on the provider, so direct
                callers and the LLM ``code_exec`` tool share the same default.
        """
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

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in this execution environment.

        Args:
            path: File path within this execution environment (relative or absolute
                  within the env's working directory).

        Returns:
            True if the file exists, False otherwise.

        Raises:
            RuntimeError: If execution environment not started.

        """
        ...

    @abstractmethod
    async def is_directory(self, path: str) -> bool:
        """Check if a path is a directory in this execution environment.

        Args:
            path: Path within this execution environment.

        Returns:
            True if the path exists and is a directory, False otherwise.

        Raises:
            RuntimeError: If execution environment not started.

        """
        ...

    @abstractmethod
    async def list_files(self, path: str) -> list[str]:
        """List all files recursively in a directory within this execution environment.

        Args:
            path: Directory path within this execution environment.

        Returns:
            List of file paths (relative to the given path) for all files in the directory.
            Returns an empty list if the path is a file or doesn't exist.

        Raises:
            RuntimeError: If execution environment not started.

        """
        ...

    def output_source_roots(self) -> tuple[OutputSourceRoot, ...]:
        """Return the absolute execution roots accepted when resolving output paths.

        Each root is an absolute path and, for directories on the host
        filesystem, carries its host path (see :class:`OutputSourceRoot`).
        Backends with absolute sandbox or container paths should override this
        so sources beneath a known root can be preserved relative to it.
        """
        return ()

    async def resolve_output_source(self, source_path: str) -> str:
        """Return the backend path that is safe to read for an output source.

        The default implementation returns the path unchanged. Backends whose
        file metadata exposes symlink targets should override this to resolve
        every component and reject sources whose target escapes an execution
        root.

        Raises:
            ValueError: If the source is invalid or escapes the execution root
                (overriding backends only).
        """
        return source_path

    async def output_destination_identity(self, destination: str, output_root: Path | str) -> str:
        """Validate a cross-environment destination and return its canonical identity."""
        normalized_destination = PurePosixPath(destination)
        normalized_root = PurePosixPath(str(output_root))
        if ".." in normalized_destination.parts:
            raise ValueError(f"Output destination escapes output directory: {destination}")
        try:
            normalized_destination.relative_to(normalized_root)
        except ValueError as exc:
            raise ValueError(f"Output destination escapes output directory: {destination}") from exc
        return normalized_destination.as_posix()

    async def save_output_files(
        self,
        paths: list[str],
        output_dir: Path | str,
        dest_env: "CodeExecToolProvider | None" = None,
    ) -> SaveOutputFilesResult:
        """Save files from this execution environment to a destination.

        Each source path is preserved relative to ``output_dir``: relative
        sources keep their directory structure, and absolute sources are made
        relative to a matching root from ``output_source_roots()``. Sources
        that are empty, contain traversal, fall outside every declared root,
        or fail ``resolve_output_source()`` are rejected, and a second source
        that resolves to an already-claimed destination is rejected rather
        than overwriting it (first requested output wins). Every rejection is
        recorded in ``SaveOutputFilesResult.failed`` as ``source_path ->
        reason``. All implementations copy files and leave sources in place.
        Host-backed providers preserve file modes.

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
        reserved_destinations: dict[str, str] = {}
        local_output_root = Path(output_dir)
        if dest_env is None:
            local_output_root.mkdir(parents=True, exist_ok=True)

        for source_path in dict.fromkeys(paths):
            try:
                relative_path = _safe_output_relative_path(
                    source_path,
                    source_roots=self.output_source_roots(),
                )
                resolved_source_path = await self.resolve_output_source(source_path)
                content = await self.read_file_bytes(resolved_source_path)

                if dest_env is not None:
                    # Transfer to another exec env (cross-environment)
                    destination = PurePosixPath(output_dir_str) / relative_path
                    identity = await dest_env.output_destination_identity(destination.as_posix(), output_dir)
                    if identity in reserved_destinations:
                        raise ValueError(
                            f"Output destination collision: {destination} "
                            f"(already claimed by {reserved_destinations[identity]})"
                        )
                    reserved_destinations[identity] = source_path
                    logger.debug(
                        "CROSS-ENV TRANSFER: %s (%d bytes) -> %s (dest_env: %s)",
                        source_path,
                        len(content),
                        destination,
                        type(dest_env).__name__,
                    )
                    await dest_env.write_file_bytes(destination.as_posix(), content)
                    result.saved.append(SavedFile(source_path, destination, len(content)))
                else:
                    # Save to local filesystem
                    destination = local_output_root.joinpath(*relative_path.parts)
                    _reserve_output_destination(
                        destination,
                        reserved_destinations,
                        claimant=source_path,
                        output_root=local_output_root,
                    )
                    logger.debug(
                        "SAVE TO LOCAL: %s (%d bytes) -> %s",
                        source_path,
                        len(content),
                        destination,
                    )
                    _atomic_write_bytes(content, destination)
                    result.saved.append(SavedFile(source_path, destination, len(content)))
            except (ValueError, FileNotFoundError) as e:
                logger.warning("TRANSFER FAILED: %s -> %s: %s", source_path, output_dir_str, e)
                result.failed[source_path] = str(e)
            except Exception as e:
                logger.exception("TRANSFER FAILED: %s -> %s", source_path, output_dir_str)
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
                    # Check if it's a directory first
                    if await source_env.is_directory(path_str):
                        # Handle directory recursively
                        # Preserve directory name when dest_dir not specified
                        dir_name = Path(path_str).name
                        files = await source_env.list_files(path_str)
                        for rel_file_path in files:
                            src_file_path = f"{path_str}/{rel_file_path}"
                            # If dest_dir specified, put files directly there
                            # Otherwise, preserve the source directory name
                            if dest_dir_str:
                                dest_path = f"{dest_dir_str}/{rel_file_path}"
                            else:
                                dest_path = f"{dir_name}/{rel_file_path}"
                            content = await source_env.read_file_bytes(src_file_path)
                            logger.debug(
                                "UPLOAD CROSS-ENV (dir): %s (%d bytes) from %s -> %s",
                                src_file_path,
                                len(content),
                                type(source_env).__name__,
                                dest_path,
                            )
                            await self.write_file_bytes(dest_path, content)
                            result.uploaded.append(UploadedFile(Path(src_file_path), dest_path, len(content)))
                    else:
                        # Single file transfer
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
            # timeout=None defers to env._shell_timeout, set in CodeExecToolProvider.__init__.
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
