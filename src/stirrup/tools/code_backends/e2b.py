"""E2B cloud execution environment backend for code execution."""

import shlex
from contextlib import AbstractAsyncContextManager
from pathlib import Path, PurePosixPath

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
    OutputSourceRoot,
    UploadedFile,
    UploadFilesResult,
)

logger = logging.getLogger(__name__)

# Home/working directory of the default E2B sandbox template. A custom template
# must use this same home directory; otherwise output-path resolution and symlink
# validation reject files that live under its actual home.
E2B_WORKING_DIR = "/home/user"


def _quote_argv_for_shell(argv: list[str]) -> str:
    """Render argv as a shell command string that runs exactly those arguments.

    Every argument is quoted. The command word is quoted even when it would
    not normally need it, so the shell cannot treat it as a reserved word
    (``time``) or an assignment — only as a plain command lookup. This matches
    the behavior of the backends that skip the shell entirely.
    """
    head = "'" + argv[0].replace("'", "'\"'\"'") + "'"
    return f"{head} {shlex.join(argv[1:])}" if len(argv) > 1 else head


def make_create_gate(max_rate: float, time_period: float = 1) -> AbstractAsyncContextManager[object]:
    """Build a shared rate-limit gate suitable for ``E2BCodeExecToolProvider(create_gate=...)``.

    Wraps ``aiolimiter.AsyncLimiter`` so callers don't have to import it directly.
    Pass a single instance to every provider that should share the same budget — e.g.
    to stay within E2B's per-account /sandboxes rate limit.

    Args:
        max_rate: Maximum number of sandbox creations allowed per ``time_period`` seconds.
        time_period: Window in seconds over which ``max_rate`` applies. Defaults to 1
                     (per-second cap), unlike ``AsyncLimiter``'s 60s default.

    Raises:
        ImportError: If the ``aiolimiter`` package is not installed. Install it via the
            ``e2b-throttle`` extra (``pip install stirrup[e2b-throttle]``) or directly
            (``pip install aiolimiter``).

    """
    try:
        from aiolimiter import AsyncLimiter
    except ImportError as e:
        raise ImportError(
            "Sandbox creation throttling requires the `aiolimiter` package, which is not installed. "
            "Install it via the `e2b-throttle` extra (e.g. `uv pip install stirrup[e2b-throttle]` "
            "or `uv add stirrup[e2b-throttle]`), or install `aiolimiter` directly.",
        ) from e
    return AsyncLimiter(max_rate=max_rate, time_period=time_period)


class E2BCodeExecToolProvider(CodeExecToolProvider):
    """E2B cloud code execution tool provider.

    Usage with Agent:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        provider = E2BCodeExecToolProvider(timeout=600, template="my-template")
        client = ChatCompletionsClient(model="gpt-5", max_tokens=8_192, context_window_tokens=64_000)
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
        create_gate: AbstractAsyncContextManager[object] | None = None,
        sandbox_kwargs: dict | None = None,
        shell_timeout: int = SHELL_TIMEOUT,
    ) -> None:
        """Initialize E2B execution environment configuration.

        Args:
            timeout: Execution environment lifetime in seconds (default: 10 minutes).
            template: Optional E2B template name/alias. A custom template must
                use ``/home/user`` as its home/working directory; output-path
                resolution and symlink validation are anchored there, so a
                template with a different home will have its output files
                rejected as outside the execution root.
            allowed_commands: Optional list of regex patterns. When set,
                             commands are matched against the patterns and
                             re-quoted so the sandbox shell runs exactly the
                             parsed arguments, with nothing expanded. When
                             None, all commands are allowed and run through
                             the shell unchanged.
            sandbox_kwargs: Additional keyword arguments forwarded to
                            ``AsyncSandbox.create`` (e.g. ``allow_internet_access=False``,
                            ``metadata={...}``, ``envs={...}``). ``timeout`` and
                            ``template`` take precedence over matching keys here.
            create_gate: Optional async context manager entered immediately before
                         each AsyncSandbox.create() call. Use to throttle/serialize
                         sandbox creation when many providers start concurrently —
                         e.g., to stay within E2B's per-account /sandboxes rate
                         limit. Use ``make_create_gate(max_rate=5)`` from this
                         module to build a shared gate, or pass any
                         ``AbstractAsyncContextManager`` of your choice.
                         ``make_create_gate`` requires the ``e2b-throttle`` extra
                         (``pip install stirrup[e2b-throttle]``).
            shell_timeout: Per-command wall-clock timeout (seconds) for every
                ``code_exec`` invocation, and the default for direct
                ``run_command`` calls that pass ``timeout=None``. Defaults to
                ``SHELL_TIMEOUT``. Distinct from ``timeout`` above, which is the
                sandbox lifetime.

        """
        super().__init__(allowed_commands=allowed_commands, shell_timeout=shell_timeout)
        self._timeout = timeout
        self._request_timeout = request_timeout
        self._template = template
        self._sandbox_kwargs = sandbox_kwargs or {}
        self._sbx: AsyncSandbox | None = None
        self._create_gate = create_gate

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        """Initialize the E2B sandbox environment and return the code_exec tool."""
        kwargs: dict = {**self._sandbox_kwargs, "timeout": self._timeout}
        if self._template:
            kwargs["template"] = self._template

        if self._create_gate is not None:
            async with self._create_gate:
                self._sbx = await AsyncSandbox.create(**kwargs)
        else:
            self._sbx = await AsyncSandbox.create(**kwargs)

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

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in the E2B sandbox.

        Args:
            path: File path within the sandbox.

        Returns:
            True if the file exists, False otherwise.

        Raises:
            RuntimeError: If environment not started.

        """
        if self._sbx is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        return await self._sbx.files.exists(path)

    async def is_directory(self, path: str) -> bool:
        """Check if a path is a directory in the E2B sandbox.

        Args:
            path: Path within the sandbox.

        Returns:
            True if the path exists and is a directory, False otherwise.

        Raises:
            RuntimeError: If environment not started.

        """
        if self._sbx is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        if not await self._sbx.files.exists(path):
            return False

        info = await self._sbx.files.get_info(path)
        return info.type == FileType.DIR

    async def list_files(self, path: str) -> list[str]:
        """List all files recursively in a directory within the E2B sandbox.

        Args:
            path: Directory path within the sandbox.

        Returns:
            List of file paths (relative to the given path) for all files in the directory.
            Returns an empty list if the path is a file or doesn't exist.

        Raises:
            RuntimeError: If environment not started.

        """
        if self._sbx is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        if not await self._sbx.files.exists(path):
            return []

        info = await self._sbx.files.get_info(path)
        if info.type != FileType.DIR:
            return []

        # Use find command to list all files recursively
        result = await self.run_command(f"find {path} -type f")
        if result.exit_code != 0:
            return []

        files = []
        for line in result.stdout.strip().split("\n"):
            if line:
                # Convert absolute path to relative path
                rel_path = line.removeprefix(f"{path}/").removeprefix(path)
                if rel_path.startswith("/"):
                    rel_path = rel_path[1:]
                if rel_path:
                    files.append(rel_path)
        return files

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        """Execute command in E2B execution environment, returning raw CommandResult.

        Args:
            cmd: Shell command to execute.
            timeout: Per-call wall-clock timeout (seconds). If None, falls back
                to ``self._shell_timeout`` set in ``__init__``.
        """
        if self._sbx is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )
        if timeout is None:
            timeout = self._shell_timeout

        # With an allowlist, run the parsed argv. E2B only accepts a command
        # string, so re-quote the argv; the sandbox shell then runs exactly
        # those arguments with nothing to expand.
        argv, rejection = self._prepare_command(cmd)
        if rejection is not None:
            return rejection
        if argv is not None:
            cmd = _quote_argv_for_shell(argv)

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

    def output_source_roots(self) -> tuple[OutputSourceRoot, ...]:
        """Return absolute execution roots accepted when resolving output paths."""
        return (OutputSourceRoot(path=E2B_WORKING_DIR),)

    async def resolve_output_source(self, source_path: str) -> str:
        """Resolve an E2B source and require a regular file inside the working root."""
        if self._sbx is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        command = f"realpath -ze -- {shlex.quote(source_path)}"
        try:
            result = await self._sbx.commands.run(
                command,
                timeout=self._shell_timeout,
                request_timeout=self._request_timeout,
            )
        except CommandExitException as exc:
            raise ValueError(f"Could not resolve E2B output source: {source_path}") from exc

        resolved_paths = [path for path in result.stdout.split("\0") if path]
        if len(resolved_paths) != 1:
            raise ValueError(f"Could not resolve E2B output source: {source_path}")
        resolved_path = PurePosixPath(resolved_paths[0])
        try:
            resolved_path.relative_to(PurePosixPath(E2B_WORKING_DIR))
        except ValueError as exc:
            raise ValueError(f"E2B output source is outside E2B execution root: {source_path}") from exc

        info = await self._sbx.files.get_info(resolved_path.as_posix())
        if info.type != FileType.FILE:
            raise ValueError(f"Path is not a file (type: {info.type})")
        return resolved_path.as_posix()

    async def output_destination_identity(self, destination: str, output_root: Path | str) -> str:
        """Validate a cross-environment destination and return its canonical identity.

        Raises:
            RuntimeError: If the execution environment is not started.
            ValueError: If the destination escapes ``output_root`` or the
                sandbox cannot canonicalize the paths.
        """
        if self._sbx is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        command = f"realpath -zm -- {shlex.quote(str(output_root))} {shlex.quote(destination)}"
        try:
            result = await self._sbx.commands.run(
                command,
                timeout=self._shell_timeout,
                request_timeout=self._request_timeout,
            )
        except CommandExitException as exc:
            raise ValueError(
                "Could not resolve cross-environment output destination: "
                f"realpath exited {exc.exit_code}: {(exc.stderr or exc.stdout).strip()}"
            ) from exc
        resolved_paths = [path for path in result.stdout.split("\0") if path]
        if len(resolved_paths) != 2:
            raise ValueError(
                "Could not resolve cross-environment output destination "
                f"(expected 2 NUL-terminated paths from realpath; stdout: {result.stdout!r}, "
                f"stderr: {result.stderr!r})"
            )

        resolved_root, resolved_destination = map(PurePosixPath, resolved_paths)
        try:
            resolved_destination.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Output destination escapes output directory: {destination}") from exc

        return resolved_destination.as_posix()

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
        dest_base = dest_dir or E2B_WORKING_DIR
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
