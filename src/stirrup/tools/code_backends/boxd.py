"""boxd cloud machine backend for code execution."""

import logging
import os
import shlex
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import anyio
from pydantic import ValidationError

try:
    from boxd import AsyncBoxd
    from boxd.errors import BoxdError, NotFoundError
    from boxd.models import ExecResult
except ImportError as e:
    raise ImportError(
        "Requires installation of the boxd extra. Install with (for example): `uv pip install stirrup[boxd]` or `uv add stirrup[boxd]`",
    ) from e

from stirrup.core.models import ImageContentBlock, Tool, ToolUseCountMetadata

from .base import (
    SHELL_TIMEOUT,
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
    OutputSourceRoot,
    UploadFilesResult,
)

logger = logging.getLogger(__name__)

# Every command, upload and output path is anchored here.
BOXD_WORKING_DIR = "/home/boxd/workspace"

# Timeout for the short internal commands that inspect paths.
_INTERNAL_COMMAND_TIMEOUT = 30

_NOT_STARTED = "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."


def _quote_argv_for_shell(argv: list[str]) -> str:
    """Render argv as a command string that runs exactly those arguments.

    The command word is quoted too, so the shell cannot read it as a reserved
    word or an assignment.
    """
    head = "'" + argv[0].replace("'", "'\"'\"'") + "'"
    return f"{head} {shlex.join(argv[1:])}" if len(argv) > 1 else head


class BoxdCodeExecToolProvider(CodeExecToolProvider):
    """Code execution in a disposable, isolated boxd machine.

    Each session creates one machine with ``isolated=True`` and deletes it on
    exit. Authenticates with ``BOXD_API_KEY``.

    Usage with Agent:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        provider = BoxdCodeExecToolProvider(region="fra")
        client = ChatCompletionsClient(model="gpt-5.6-luna", max_tokens=8_192, context_window_tokens=1_000_000)
        agent = Agent(client=client, name="assistant", tools=[provider])
        async with agent.session() as session:
            await session.run("Run Python code")

    Standalone usage:
        provider = BoxdCodeExecToolProvider(allowed_commands=[r"^python", r"^pip"])
        async with provider as tool:
            result = await provider.run_command("python script.py")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        org: str | None = None,
        region: str | None = None,
        vcpu: int | None = None,
        memory: int | str | None = None,
        env: Mapping[str, str] | None = None,
        labels: Mapping[str, str] | None = None,
        name_prefix: str = "stirrup-",
        allowed_commands: list[str] | None = None,
        shell_timeout: int = SHELL_TIMEOUT,
    ) -> None:
        """Initialize the boxd machine configuration.

        Args:
            api_key: boxd API key. Defaults to the ``BOXD_API_KEY`` environment
                variable, or a ``BOXD_TOKEN`` token. Entering the session without
                any of them raises ``ValueError``.
            org: Organization that owns the machine. Defaults to ``BOXD_ORG``, then
                the key's organization.
            region: Region to create the machine in, such as ``"fra"`` or ``"nyc"``.
            vcpu: Number of vCPUs. Set together with ``memory``.
            memory: Memory size, such as ``"8G"``. Set together with ``vcpu``.
            env: Environment variables passed to every ``run_command`` call. The
                code the agent runs can read them.
            labels: Labels to set on the machine.
            name_prefix: Prefix for the generated machine name.
            allowed_commands: Optional list of regex patterns. When set, commands
                are matched against the patterns and re-quoted so the machine's
                shell runs exactly the parsed arguments, with nothing expanded.
                When None, all commands run through bash.
            shell_timeout: Per-command wall-clock timeout (seconds) for every
                ``code_exec`` invocation, and the default for direct
                ``run_command`` calls that pass ``timeout=None``.

        """
        super().__init__(allowed_commands=allowed_commands, shell_timeout=shell_timeout)
        self._api_key = api_key
        self._org = org
        self._region = region
        self._vcpu = vcpu
        self._memory = memory
        self._env = dict(env) if env else None
        self._labels = dict(labels) if labels else None
        self._name_prefix = name_prefix
        self._client: AsyncBoxd | None = None
        self._machine_id: str | None = None

    @property
    def machine_id(self) -> str | None:
        """Return the id of the running machine, if any."""
        return self._machine_id

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        """Create the isolated machine and return the code_exec tool."""
        if self._client is not None:
            raise RuntimeError("This provider already has a running machine.")
        if not (self._api_key or os.environ.get("BOXD_API_KEY") or os.environ.get("BOXD_TOKEN")):
            raise ValueError("boxd credentials not found. Set BOXD_API_KEY or pass api_key=.")

        client = AsyncBoxd(api_key=self._api_key, region=self._region)
        name = f"{self._name_prefix}{uuid.uuid4().hex[:12]}"
        machine_id: str | None = None
        try:
            machine = await client.machines.create(
                name,
                org=self._org or os.environ.get("BOXD_ORG"),
                isolated=True,
                labels=self._labels,
                vcpu=self._vcpu,
                memory=self._memory,
                # Keep the machine out of standby while a CPU-only command runs without network traffic.
                auto_suspend_timeout=0,
                region=self._region,
            )
            machine_id = machine.id
            await client.machines.wait_until_ready(machine_id)
            setup = await client.machines.exec(
                machine_id,
                ["mkdir", "-p", BOXD_WORKING_DIR],
                timeout=_INTERNAL_COMMAND_TIMEOUT,
            )
            if setup.exit_code != 0:
                raise RuntimeError(f"Could not create {BOXD_WORKING_DIR}: {setup.stderr}")
        except BaseException:
            with anyio.CancelScope(shield=True):
                await self._release(client, machine_id or await self._find_by_name(client, name))
            raise

        self._client = client
        self._machine_id = machine_id
        return self.get_code_exec_tool()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Delete the machine, also when the session is being cancelled."""
        client, machine_id = self._client, self._machine_id
        self._client = None
        self._machine_id = None
        if client is not None:
            with anyio.CancelScope(shield=True):
                await self._release(client, machine_id)

    @staticmethod
    async def _find_by_name(client: AsyncBoxd, name: str) -> str | None:
        """Find a machine whose create call failed after the server accepted it."""
        try:
            machine = await client.machines.get(name)
        except NotFoundError:
            return None
        except Exception:
            logger.exception("Could not check for boxd machine %s after a failed create", name)
            return None
        return machine.id if machine.name == name else None

    @staticmethod
    async def _release(client: AsyncBoxd, machine_id: str | None) -> None:
        """Delete the machine and close the client without masking an earlier error."""
        try:
            if machine_id is not None:
                await client.machines.delete(machine_id)
        except Exception:
            logger.exception("Failed to delete boxd machine %s", machine_id)
        finally:
            await client.close()

    def _require_started(self) -> tuple[AsyncBoxd, str]:
        if self._client is None or self._machine_id is None:
            raise RuntimeError(_NOT_STARTED)
        return self._client, self._machine_id

    @staticmethod
    def _absolute(path: str) -> str:
        """Anchor a relative path at the working directory."""
        return (PurePosixPath(BOXD_WORKING_DIR) / path).as_posix()

    async def _inspect(self, argv: Sequence[str]) -> ExecResult:
        """Run a short internal command in the working directory."""
        client, machine_id = self._require_started()
        return await client.machines.exec(
            machine_id,
            list(argv),
            cwd=BOXD_WORKING_DIR,
            timeout=_INTERNAL_COMMAND_TIMEOUT,
        )

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        """Execute a command in the machine, returning a raw CommandResult.

        Args:
            cmd: Shell command to execute.
            timeout: Per-call wall-clock timeout (seconds). If None, falls back
                to ``self._shell_timeout`` set in ``__init__``.
        """
        client, machine_id = self._require_started()
        if timeout is None:
            timeout = self._shell_timeout

        argv, rejection = self._prepare_command(cmd)
        if rejection is not None:
            return rejection
        command: str | list[str] = _quote_argv_for_shell(argv) if argv is not None else ["bash", "-c", cmd]

        try:
            result = await client.machines.exec(
                machine_id,
                command,
                env=self._env,
                cwd=BOXD_WORKING_DIR,
                timeout=timeout,
            )
        except BoxdError as exc:
            if str(exc).startswith("exec timed out"):
                return CommandResult(exit_code=1, stdout="", stderr=str(exc), error_kind="timeout")
            # Not retried: the command may already have run.
            return CommandResult(exit_code=1, stdout="", stderr=str(exc), error_kind="execution_error")

        return CommandResult(exit_code=result.exit_code, stdout=str(result.stdout), stderr=str(result.stderr))

    async def read_file_bytes(self, path: str) -> bytes:
        """Read file content as bytes from the machine.

        Args:
            path: File path, relative to the working directory or absolute.

        Returns:
            File contents as bytes.

        Raises:
            RuntimeError: If environment not started.
            FileNotFoundError: If file does not exist.

        """
        client, machine_id = self._require_started()
        try:
            return await client.machines.files.download(machine_id, self._absolute(path))
        except NotFoundError as exc:
            raise FileNotFoundError(f"File not found: {path}") from exc

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        """Write bytes to a file in the machine, creating parent directories.

        Args:
            path: Destination path, relative to the working directory or absolute.
            content: File contents to write.

        Raises:
            RuntimeError: If environment not started.

        """
        client, machine_id = self._require_started()
        await client.machines.files.upload(machine_id, self._absolute(path), content)

    async def file_exists(self, path: str) -> bool:
        """Check if a path exists in the machine."""
        return (await self._inspect(["test", "-e", self._absolute(path)])).exit_code == 0

    async def is_directory(self, path: str) -> bool:
        """Check if a path is a directory in the machine."""
        return (await self._inspect(["test", "-d", self._absolute(path)])).exit_code == 0

    async def list_files(self, path: str) -> list[str]:
        """List all files recursively in a directory within the machine.

        Args:
            path: Directory path in the machine.

        Returns:
            File paths relative to ``path``. Empty if the path is a file or doesn't exist.

        """
        if not await self.is_directory(path):
            return []
        result = await self._inspect(["find", self._absolute(path), "-type", "f", "-printf", "%P\\0"])
        if result.exit_code != 0:
            return []
        return [name for name in str(result.stdout).split("\0") if name]

    def output_source_roots(self) -> tuple[OutputSourceRoot, ...]:
        """Return absolute execution roots accepted when resolving output paths."""
        return (OutputSourceRoot(path=BOXD_WORKING_DIR),)

    async def resolve_output_source(self, source_path: str) -> str:
        """Resolve a source and require a regular file inside the working directory."""
        result = await self._inspect(["realpath", "-ze", "--", self._absolute(source_path)])
        resolved_paths = [path for path in str(result.stdout).split("\0") if path]
        if result.exit_code != 0 or len(resolved_paths) != 1:
            raise ValueError(f"Could not resolve boxd output source: {source_path}")

        resolved_path = PurePosixPath(resolved_paths[0])
        try:
            resolved_path.relative_to(PurePosixPath(BOXD_WORKING_DIR))
        except ValueError as exc:
            raise ValueError(f"boxd output source is outside the working directory: {source_path}") from exc

        if (await self._inspect(["test", "-f", resolved_path.as_posix()])).exit_code != 0:
            raise ValueError(f"Path is not a file: {source_path}")
        return resolved_path.as_posix()

    async def output_destination_identity(self, destination: str, output_root: Path | str) -> str:
        """Validate a cross-environment destination and return its canonical identity.

        Raises:
            RuntimeError: If the execution environment is not started.
            ValueError: If the destination escapes ``output_root`` or the
                machine cannot canonicalize the paths.
        """
        result = await self._inspect(
            ["realpath", "-zm", "--", self._absolute(str(output_root)), self._absolute(destination)]
        )
        resolved_paths = [path for path in str(result.stdout).split("\0") if path]
        if result.exit_code != 0 or len(resolved_paths) != 2:
            raise ValueError(f"Could not resolve cross-environment output destination: {destination}")

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
        """Upload files to the machine.

        A local directory uploaded without ``dest_dir`` keeps its name under the
        working directory. Everything else uses the base class implementation.

        Args:
            *paths: File or directory paths to upload.
            source_env: If provided, paths are within source_env.
            dest_dir: Destination directory in the machine.

        Returns:
            UploadFilesResult containing lists of uploaded files and any failures.

        """
        self._require_started()
        if source_env is not None or dest_dir is not None:
            return await super().upload_files(*paths, source_env=source_env, dest_dir=dest_dir)

        result = UploadFilesResult()
        for path in paths:
            local_path = Path(path)
            directory = local_path.resolve().name if local_path.is_dir() else None
            uploaded = await super().upload_files(path, dest_dir=directory)
            result.uploaded.extend(uploaded.uploaded)
            result.failed.update(uploaded.failed)
        return result

    async def view_image(self, path: str) -> ImageContentBlock:
        """Read and return an image file from the machine.

        Args:
            path: Path to image file in the machine.

        Returns:
            ImageContentBlock containing the image data.

        Raises:
            RuntimeError: If execution environment not started.
            FileNotFoundError: If file does not exist.
            ValueError: If the file is not a supported, valid image.

        """
        if not path.lower().endswith((".png", ".jpg", ".jpeg")):
            raise ValueError(f"Unsupported image type for `{path}`. Only .png, .jpg, or .jpeg are allowed.")

        try:
            content = await self.read_file_bytes(path)
        except BoxdError as e:
            raise ValueError(f"Could not read image `{path}`: {e}") from e
        try:
            return ImageContentBlock(data=content)
        except (ValidationError, SyntaxError, OSError) as e:
            # Pillow reports damaged files with SyntaxError or OSError, not only ValidationError.
            raise ValueError("You submitted a corrupt/unsupported image file.") from e
