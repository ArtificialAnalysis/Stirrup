"""Daytona cloud execution environment backend for code execution."""

import math
import shlex
import uuid
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

try:
    from daytona import (
        AsyncDaytona,
        AsyncSandbox,
        CreateSandboxFromImageParams,
        CreateSandboxFromSnapshotParams,
        DaytonaConfig,
        DaytonaError,
        DaytonaNotFoundError,
        DaytonaTimeoutError,
        DaytonaValidationError,
        FileUpload,
        Image,
        SessionExecuteRequest,
    )
except ImportError as e:
    raise ImportError(
        "Requires installation of the daytona extra. Install with (for example): "
        "`uv pip install stirrup[daytona]` or `uv add stirrup[daytona]`",
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


def _quote_argv_for_shell(argv: list[str]) -> str:
    """Render argv as a shell command string that runs exactly those arguments.

    Every argument is quoted. The command word is quoted even when it would
    not normally need it, so the shell cannot treat it as a reserved word
    (``time``) or an assignment — only as a plain command lookup. This matches
    the behavior of the backends that skip the shell entirely.
    """
    head = "'" + argv[0].replace("'", "'\"'\"'") + "'"
    return f"{head} {shlex.join(argv[1:])}" if len(argv) > 1 else head


class DaytonaCodeExecToolProvider(CodeExecToolProvider):
    """Daytona cloud code execution tool provider.

    Credentials are read from the ``DAYTONA_API_KEY``, ``DAYTONA_API_URL``, and
    ``DAYTONA_TARGET`` environment variables unless an explicit ``config`` is
    provided.

    Commands run through per-command sandbox sessions, so stdout and stderr are
    captured separately and no shell state (working directory, variables) leaks
    between ``code_exec`` invocations.

    Usage with Agent:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        provider = DaytonaCodeExecToolProvider(timeout=600, snapshot="my-snapshot")
        client = ChatCompletionsClient(model="gpt-5.6-luna", max_tokens=8_192, context_window_tokens=1_000_000)
        agent = Agent(client=client, name="assistant", tools=[provider])
        async with agent.session() as session:
            await session.run("Run Python code")

    Standalone usage:
        provider = DaytonaCodeExecToolProvider(allowed_commands=[r"^python", r"^pip"])
        async with provider as tool:
            result = await provider.run_command("python script.py")
    """

    def __init__(
        self,
        *,
        timeout: int = SANDBOX_TIMEOUT,
        request_timeout: int = SANDBOX_REQUEST_TIMEOUT,
        snapshot: str | None = None,
        image: "str | Image | None" = None,
        auto_stop_interval: int = 0,
        config: "DaytonaConfig | None" = None,
        allowed_commands: list[str] | None = None,
        sandbox_kwargs: dict | None = None,
        shell_timeout: int = SHELL_TIMEOUT,
    ) -> None:
        """Initialize Daytona execution environment configuration.

        Args:
            timeout: Execution environment lifetime in seconds (default: 10
                minutes). Converted to Daytona's wall-clock ``ttl_minutes``
                (rounded up to whole minutes; the API only accepts integer
                minutes), after which the sandbox is destroyed regardless of
                state.
            request_timeout: Timeout in seconds for individual API operations
                (sandbox creation, file transfers).
            snapshot: Optional Daytona snapshot name to create the sandbox
                from. When neither ``snapshot`` nor ``image`` is given, the
                default Daytona snapshot is used.
            image: Optional image (name or declarative ``Image``) to create
                the sandbox from. Mutually exclusive with ``snapshot``.
            auto_stop_interval: Inactivity interval in minutes after which
                Daytona stops the sandbox. Defaults to 0 (disabled) so the
                sandbox lives for exactly ``timeout``; set a positive value
                to re-enable Daytona's native inactivity auto-stop.
            config: Optional ``DaytonaConfig`` with explicit ``api_key``,
                ``api_url``, and ``target``. When None, the SDK reads the
                ``DAYTONA_*`` environment variables.
            allowed_commands: Optional list of regex patterns. When set,
                             commands are matched against the patterns and
                             re-quoted so the sandbox shell runs exactly the
                             parsed arguments, with nothing expanded. When
                             None, all commands are allowed and run through
                             the shell unchanged.
            sandbox_kwargs: Additional keyword arguments forwarded to the
                            sandbox creation params (e.g. ``env_vars={...}``,
                            ``labels={...}``, ``ephemeral=True``,
                            ``resources=Resources(cpu=2)``). Explicit arguments
                            (``snapshot``, ``image``, ``auto_stop_interval``,
                            and the ``ttl_minutes`` derived from ``timeout``)
                            take precedence over matching keys here.
            shell_timeout: Per-command wall-clock timeout (seconds) for every
                ``code_exec`` invocation, and the default for direct
                ``run_command`` calls that pass ``timeout=None``. Defaults to
                ``SHELL_TIMEOUT``. Distinct from ``timeout`` above, which is the
                sandbox lifetime.

        Raises:
            ValueError: If both ``snapshot`` and ``image`` are provided.

        """
        super().__init__(allowed_commands=allowed_commands, shell_timeout=shell_timeout)
        if snapshot is not None and image is not None:
            raise ValueError("Provide either `snapshot` or `image`, not both.")
        self._timeout = timeout
        self._request_timeout = request_timeout
        self._snapshot = snapshot
        self._image = image
        self._auto_stop_interval = auto_stop_interval
        self._config = config
        self._sandbox_kwargs = sandbox_kwargs or {}
        self._daytona: AsyncDaytona | None = None
        self._sbx: AsyncSandbox | None = None
        self._work_dir: str | None = None

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        """Initialize the Daytona sandbox environment and return the code_exec tool."""
        kwargs: dict = {
            **self._sandbox_kwargs,
            "ttl_minutes": max(1, math.ceil(self._timeout / 60)),
            "auto_stop_interval": self._auto_stop_interval,
        }
        if self._image is not None:
            params: CreateSandboxFromImageParams | CreateSandboxFromSnapshotParams = CreateSandboxFromImageParams(
                image=self._image, **kwargs
            )
        else:
            if self._snapshot is not None:
                kwargs["snapshot"] = self._snapshot
            params = CreateSandboxFromSnapshotParams(**kwargs)

        self._daytona = AsyncDaytona(self._config)
        try:
            self._sbx = await self._daytona.create(params, timeout=float(self._request_timeout))
            # The working directory is snapshot/image-dependent; resolve it once so
            # output-path validation is anchored at the actual execution root.
            self._work_dir = await self._sbx.get_work_dir()
        except BaseException:
            await self._cleanup()
            raise

        return self.get_code_exec_tool()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Cleanup the Daytona execution environment."""
        await self._cleanup()

    async def _cleanup(self) -> None:
        """Delete the sandbox (if any) and close the Daytona client."""
        try:
            if self._sbx is not None:
                try:
                    await self._sbx.delete()
                except DaytonaNotFoundError:
                    logger.debug("Daytona sandbox already deleted.")
        finally:
            self._sbx = None
            self._work_dir = None
            if self._daytona is not None:
                await self._daytona.close()
                self._daytona = None

    def _require_sandbox(self) -> AsyncSandbox:
        """Return the active sandbox or raise if the environment is not started."""
        if self._sbx is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )
        return self._sbx

    async def _run_in_session(self, cmd: str, *, timeout: int) -> CommandResult:
        """Run one command in a fresh sandbox session and return its raw result.

        Each command gets its own session so stdout and stderr are captured
        separately while shell state still cannot leak between commands. The
        session is always torn down afterwards, which also kills the command
        if it outlived ``timeout``.

        Runs ``cmd`` exactly as given, with no allowlist check. Only pass
        trusted, internally built commands here; model-generated commands
        must go through ``run_command``, which enforces the allowlist.

        The command is wrapped in a subshell because session commands run in
        the session's own shell: a bare ``exit`` would otherwise terminate the
        session before its exit code and output can be reported. The trailing
        newline keeps a final ``# comment`` from swallowing the closing paren.
        """
        sbx = self._require_sandbox()
        session_id = f"stirrup-{uuid.uuid4().hex}"
        await sbx.process.create_session(session_id)
        try:
            response = await sbx.process.execute_session_command(
                session_id,
                SessionExecuteRequest(command=f"( {cmd}\n)", run_async=False),
                timeout=timeout,
            )
            return CommandResult(
                exit_code=response.exit_code if response.exit_code is not None else 0,
                stdout=response.stdout or "",
                stderr=response.stderr or "",
            )
        finally:
            try:
                await sbx.process.delete_session(session_id)
            except DaytonaError as exc:
                logger.debug("Failed to delete Daytona session %s: %s", session_id, exc)

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        """Execute command in Daytona execution environment, returning raw CommandResult.

        Args:
            cmd: Shell command to execute.
            timeout: Per-call wall-clock timeout (seconds). If None, falls back
                to ``self._shell_timeout`` set in ``__init__``.
        """
        self._require_sandbox()
        if timeout is None:
            timeout = self._shell_timeout

        # With an allowlist, run the parsed argv. Daytona only accepts a command
        # string, so re-quote the argv; the sandbox shell then runs exactly
        # those arguments with nothing to expand.
        argv, rejection = self._prepare_command(cmd)
        if rejection is not None:
            return rejection
        if argv is not None:
            cmd = _quote_argv_for_shell(argv)

        try:
            return await self._run_in_session(cmd, timeout=timeout)
        except DaytonaValidationError as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="invalid_argument",
                advice="Avoid NUL/control bytes and nested 'bash -lc'; use a quoted heredoc or base64 write.",
            )
        except DaytonaTimeoutError as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="timeout",
            )

    async def read_file_bytes(self, path: str) -> bytes:
        """Read file content as bytes from the Daytona sandbox.

        Args:
            path: File path within the sandbox.

        Returns:
            File contents as bytes.

        Raises:
            RuntimeError: If environment not started.
            FileNotFoundError: If file does not exist.

        """
        sbx = self._require_sandbox()

        if not await self.file_exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        # `download_file` dispatches on positional args; the timeout must stay positional.
        file_bytes = await sbx.fs.download_file(path, self._request_timeout)
        return bytes(file_bytes)

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        """Write bytes to a file in the Daytona sandbox.

        Args:
            path: Destination path within the sandbox.
            content: File contents to write.

        Raises:
            RuntimeError: If environment not started.

        """
        sbx = self._require_sandbox()
        await sbx.fs.upload_file(content, path, timeout=self._request_timeout)

    async def file_exists(self, path: str) -> bool:
        """Check if a file exists in the Daytona sandbox.

        Args:
            path: File path within the sandbox.

        Returns:
            True if the file exists, False otherwise.

        Raises:
            RuntimeError: If environment not started.

        """
        sbx = self._require_sandbox()
        try:
            await sbx.fs.get_file_info(path)
        except DaytonaNotFoundError:
            return False
        return True

    async def is_directory(self, path: str) -> bool:
        """Check if a path is a directory in the Daytona sandbox.

        Args:
            path: Path within the sandbox.

        Returns:
            True if the path exists and is a directory, False otherwise.

        Raises:
            RuntimeError: If environment not started.

        """
        sbx = self._require_sandbox()
        try:
            info = await sbx.fs.get_file_info(path)
        except DaytonaNotFoundError:
            return False
        return bool(info.is_dir)

    async def list_files(self, path: str) -> list[str]:
        """List all files recursively in a directory within the Daytona sandbox.

        Args:
            path: Directory path within the sandbox.

        Returns:
            List of file paths (relative to the given path) for all files in the directory.
            Returns an empty list if the path is a file or doesn't exist.

        Raises:
            RuntimeError: If environment not started.

        """
        self._require_sandbox()

        if not await self.is_directory(path):
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

    def output_source_roots(self) -> tuple[OutputSourceRoot, ...]:
        """Return absolute execution roots accepted when resolving output paths."""
        if self._work_dir is None:
            return ()
        return (OutputSourceRoot(path=self._work_dir),)

    async def _canonicalize_path(self, path: str, *, require_exists: bool) -> str | None:
        """Resolve one path with realpath inside the sandbox, or None on failure.

        Parses newline-terminated output instead of ``realpath -z``: the
        session transport is not NUL-safe on every snapshot/image (some strip
        NUL bytes from stdout), while realpath always terminates its single
        result with exactly one newline. Resolving one path per call means
        the output never needs splitting, so even filenames containing
        newlines survive.
        """
        flag = "-e" if require_exists else "-m"
        result = await self._run_in_session(
            f"realpath {flag} -- {shlex.quote(path)}",
            timeout=self._shell_timeout,
        )
        if result.exit_code != 0 or not result.stdout.endswith("\n"):
            return None
        return result.stdout[:-1] or None

    async def resolve_output_source(self, source_path: str) -> str:
        """Resolve a Daytona source and require a regular file inside the working root."""
        sbx = self._require_sandbox()
        if self._work_dir is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        resolved = await self._canonicalize_path(source_path, require_exists=True)
        if resolved is None:
            raise ValueError(f"Could not resolve Daytona output source: {source_path}")
        resolved_path = PurePosixPath(resolved)
        try:
            resolved_path.relative_to(PurePosixPath(self._work_dir))
        except ValueError as exc:
            raise ValueError(f"Daytona output source is outside Daytona execution root: {source_path}") from exc

        info = await sbx.fs.get_file_info(resolved_path.as_posix())
        if info.is_dir:
            raise ValueError("Path is not a file (is a directory)")
        return resolved_path.as_posix()

    async def output_destination_identity(self, destination: str, output_root: Path | str) -> str:
        """Validate a cross-environment destination and return its canonical identity.

        Raises:
            RuntimeError: If the execution environment is not started.
            ValueError: If the destination escapes ``output_root`` or the
                sandbox cannot canonicalize the paths.
        """
        self._require_sandbox()

        resolved_root = await self._canonicalize_path(str(output_root), require_exists=False)
        resolved_destination = await self._canonicalize_path(destination, require_exists=False)
        if resolved_root is None or resolved_destination is None:
            raise ValueError(f"Could not resolve cross-environment output destination: {destination}")

        try:
            PurePosixPath(resolved_destination).relative_to(PurePosixPath(resolved_root))
        except ValueError as exc:
            raise ValueError(f"Output destination escapes output directory: {destination}") from exc

        return PurePosixPath(resolved_destination).as_posix()

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: "CodeExecToolProvider | None" = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        """Upload files to the Daytona sandbox.

        When source_env is None (local filesystem), all files are sent in a
        single batched ``fs.upload_files()`` request that streams each file
        from disk, so large uploads are not buffered into memory.
        Directories are uploaded recursively, preserving their structure.
        Because the batch is one request, a transfer failure reports every
        planned file as failed.

        When source_env is provided (cross-environment transfer), files are copied
        using the base class implementation via read/write primitives.

        Args:
            *paths: File or directory paths to upload. If source_env is None, these
                    are local filesystem paths. If source_env is provided, these are
                    paths within source_env.
            source_env: If provided, paths are within source_env. If None, paths are
                        local filesystem paths.
            dest_dir: Destination directory in the sandbox.
                      If None, files are placed in the sandbox working directory.

        Returns:
            UploadFilesResult containing lists of uploaded files and any failures.

        """
        sbx = self._require_sandbox()

        # If source_env is provided, use the base class implementation (cross-env transfer)
        if source_env is not None:
            return await super().upload_files(*paths, source_env=source_env, dest_dir=dest_dir)

        # Local filesystem - use the batched Daytona API
        dest_base = dest_dir or self._work_dir
        if dest_base is None:
            raise RuntimeError("ExecutionEnvironment not started.")
        result = UploadFilesResult()

        planned: list[UploadedFile] = []
        for source in paths:
            original_name = Path(source).name  # Get name BEFORE resolve
            source = Path(source).resolve()

            if not source.exists():
                result.failed[str(source)] = "File or directory does not exist"
                logger.warning("Upload source does not exist: %s", source)
                continue

            if source.is_file():
                planned.append(
                    UploadedFile(
                        source_path=source,
                        dest_path=f"{dest_base}/{original_name}",
                        size=source.stat().st_size,
                    ),
                )
            elif source.is_dir():
                # Upload all files in directory recursively
                # If dest_dir was explicitly provided, copy contents directly to dest_base
                # Otherwise, create a subdirectory with the source's name
                for file_path in source.rglob("*"):
                    if file_path.is_file():
                        relative = file_path.relative_to(source)
                        dest = f"{dest_base}/{relative}" if dest_dir else f"{dest_base}/{source.name}/{relative}"
                        planned.append(
                            UploadedFile(
                                source_path=file_path,
                                dest_path=dest,
                                size=file_path.stat().st_size,
                            ),
                        )

        if not planned:
            return result

        uploads = [FileUpload(source=str(file.source_path), destination=file.dest_path) for file in planned]
        try:
            await sbx.fs.upload_files(uploads, timeout=self._request_timeout)
        except Exception as exc:
            for file in planned:
                result.failed[str(file.source_path)] = str(exc)
            logger.exception("Failed to upload files to Daytona sandbox")
            return result

        result.uploaded.extend(planned)
        for file in planned:
            logger.debug("Uploaded file: %s -> %s", file.source_path, file.dest_path)
        return result

    async def view_image(self, path: str) -> ImageContentBlock:
        """Read and return an image file from the Daytona execution environment.

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
