"""Backgroundable code execution: start commands, return while they run, and poll or stop them later.

``BackgroundCodeExecToolProvider`` wraps a sandboxed ``CodeExecToolProvider`` (Docker, E2B, or your own)
and replaces its blocking ``code_exec`` tool with two tools:

- ``code_exec`` starts a Bash command and waits up to ``yield_time_ms``. A command still running when the
  wait ends keeps running, and the result carries a process ID.
- ``process_control`` polls, terminates, or lists those processes.

Each command runs in its own tmux session inside the sandbox. Its combined stdout and stderr go to a
bounded ring buffer, and every result saves the unread output to a file and previews its head and tail.
On exit the provider terminates every process it started.

The sandbox needs Linux, tmux, and procps; see ``BACKGROUND_TOOLS`` for the full list.
"""

from __future__ import annotations

import base64
import binascii
import logging
import secrets
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, assert_never
from xml.sax.saxutils import quoteattr

import anyio
from pydantic import BaseModel, ConfigDict, Field, model_validator

from stirrup.core.models import ImageContentBlock, Tool, ToolResult, ToolUseCountMetadata
from stirrup.tools.code_backends.base import (
    CodeExecToolProvider,
    CommandResult,
    OutputSourceRoot,
    SaveOutputFilesResult,
    UploadFilesResult,
)
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

_ENVIRONMENT_PROBE_COMMAND = "echo"
_ENVIRONMENT_PROBE_TIMEOUT_SEC = 10
_CONTROL_TIMEOUT_SEC = 3 * 60
BACKGROUND_TOOLS = (
    "awk",
    "bash",
    "base64",
    "cat",
    "dd",
    "dirname",
    "head",
    "mkdir",
    "mv",
    "pgrep",
    "ps",
    "rm",
    "rmdir",
    "setsid",
    "sleep",
    "tail",
    "tmux",
    "touch",
    "tr",
    "wc",
)
_BACKGROUND_TOOL_NAMES = " ".join(BACKGROUND_TOOLS)
_PROBE_BACKGROUND_TOOLS_COMMAND = f"""
missing=""
for bin in {_BACKGROUND_TOOL_NAMES}; do
  command -v "$bin" >/dev/null 2>&1 || missing="$missing $bin"
done
if [ -n "$missing" ]; then
  echo "missing:$missing"
  exit 1
fi
"""
_CGROUP_FS = "/sys/fs/cgroup"
_CGROUP_PROBE_COMMAND = """
test -f /sys/fs/cgroup/cgroup.controllers || exit 1
rel=$(awk -F: '$1==0 {print $3}' /proc/self/cgroup)
name=stirrup_probe_$$
try() {
  parent=$1
  test -d "$parent" || return 1
  cg="$parent/$name"
  mkdir "$cg" 2>/dev/null || return 1
  if printf '%s\\n' "$$" > "$cg/cgroup.procs" 2>/dev/null && test -w "$cg/cgroup.kill"; then
    printf '%s\\n' "$$" > "$parent/cgroup.procs" 2>/dev/null || true
    rmdir "$cg" 2>/dev/null || true
    printf '%s\\n' "$parent"
    return 0
  fi
  printf '%s\\n' "$$" > "$parent/cgroup.procs" 2>/dev/null || true
  rmdir "$cg" 2>/dev/null || true
  return 1
}
path=$rel
while :; do
  try /sys/fs/cgroup$path && exit 0
  test -n "$path" || break
  case "$path" in */*) path=${path%/*} ;; *) path= ;; esac
done
exit 1
"""
# Sandboxes cap a single exec argument (often near 128 KiB); longer scripts go in chunks.
_INLINE_COMMAND_MAX_BYTES = 60 * 1024
_COMMAND_CHUNK_CHARS = 48 * 1024
_PROCESS_ROOT_BASE = "/tmp/stirrup/background"
_PROCESS_ID_PREFIX = "proc_"
_SESSION_PREFIX = "stirrup_"
_PROCESS_POLL_INITIAL_SEC = 0.05
_PROCESS_POLL_MAX_SEC = 5.0
_TERMINATE_GRACE_SEC = 1
_CHECK_START_MARKER = "stirrup_check_start"
_CHECK_COMPLETE_MARKER = "stirrup_check_complete"
_CHECK_RETRY_DELAYS_SEC = (1, 2, 4, 8)
_PROCESS_ID_BYTES = 4
_LIST_COMMAND_PREVIEW_CHARS = 80

DEFAULT_COMMAND_TIMEOUT_SEC = 5 * 60
DEFAULT_MAX_YIELD_TIME_SEC = 20 * 60
# Models trained on other harnesses poll with very short waits; a floor caps that turn cost.
DEFAULT_MIN_WAIT_SEC = 10.0
DEFAULT_OUTPUT_SPILL_THRESHOLD_CHARS = 30_000
DEFAULT_OUTPUT_SPILL_MAX_BYTES = 64 * 1024 * 1024
# stdin -> $1 as a $3-byte ring; total bytes written -> $2
_RING_CAPTURE_SCRIPT = """\
log=$1
written_path=$2
max=$3
chunk=$(dirname "$log")/.chunk
printf '0\\n' > "$written_path"
: > "$log"
written=0
while true; do
  dd if=/dev/stdin of="$chunk" bs=65536 count=1 2>/dev/null || true
  n=$(($(wc -c < "$chunk")))
  [ "$n" -eq 0 ] && break
  i=0
  remaining=$n
  while [ "$remaining" -gt 0 ]; do
    pos=$((written % max))
    space=$((max - pos))
    take=$remaining
    [ "$take" -gt "$space" ] && take=$space
    tail -c +$((i + 1)) "$chunk" | head -c "$take" | \
      dd of="$log" conv=notrunc oflag=seek_bytes seek="$pos" bs=65536 2>/dev/null
    written=$((written + take))
    i=$((i + take))
    remaining=$((remaining - take))
  done
  printf '%s\\n' "$written" > "${written_path}.tmp"
  mv "${written_path}.tmp" "$written_path"
done
rm -f "$chunk"
"""
_logger = logging.getLogger(__name__)


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _cd_command(workdir: str) -> str:
    """Build a `cd` that lets bash expand a leading `~` and fails when `~` expands to nothing."""
    if workdir == "~":
        target = "~"
    elif workdir.startswith("~/"):
        target = "~/" + shlex.quote(workdir[2:])
    else:
        return f"cd -- {shlex.quote(workdir)}"
    empty_home = f"printf 'cd: %s: ~ expands to an empty path\\n' {shlex.quote(workdir)} >&2; false"
    return f"case ~ in '') {empty_home};; *) cd -- {target};; esac"


def _fits_inline(command: str) -> bool:
    return len(command.encode()) <= _INLINE_COMMAND_MAX_BYTES


def _format_output_size(size: int) -> str:
    if size >= 1024 * 1024 and size % (1024 * 1024) == 0:
        return f"{size // (1024 * 1024)}MB"
    if size >= 1024 and size % 1024 == 0:
        return f"{size // 1024}KB"
    if size < 1_000:
        return f"{size}B"
    if size < 1_000_000:
        return f"{size / 1_000:.1f}KB"
    return f"{size / 1_000_000:.1f}MB"


class ProcessOperationError(RuntimeError):
    """A process operation failed while the environment stayed available."""


type WaitFor = Literal["exit", "output"]
type ProcessAction = Literal["poll", "terminate", "list"]
type ProcessStatus = Literal["running", "completed", "terminated"]
type ReturnReason = Literal["exit", "output", "yield_expired", "no_wait"]


class BackgroundableCodeExecutionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cmd: Annotated[
        str,
        Field(description="Bash command to start. An empty string only waits for yield_time_ms."),
    ]
    yield_time_ms: Annotated[
        float,
        Field(description=f"Milliseconds to wait. Defaults to {DEFAULT_MIN_WAIT_SEC * 1000:g}."),
    ] = DEFAULT_MIN_WAIT_SEC * 1000
    wait_for: Annotated[
        WaitFor,
        Field(
            description=(
                "'exit' waits for the command to exit. 'output' also returns on new stdout or stderr "
                "after the minimum wait."
            )
        ),
    ] = "exit"
    workdir: Annotated[
        str | None,
        Field(description="Directory in which to run the command. Defaults to the environment's working directory."),
    ] = None


class ProcessControlParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Annotated[
        ProcessAction,
        Field(
            description=("'poll' waits for a process, 'terminate' stops it, and 'list' returns only running processes.")
        ),
    ]
    process_id: Annotated[
        str | None,
        Field(description="ID returned by code_exec. Required for poll and terminate; ignored when action is list."),
    ] = None
    yield_time_ms: Annotated[
        float,
        Field(
            description=(
                f"Milliseconds to wait. Defaults to {DEFAULT_MIN_WAIT_SEC * 1000:g}; ignored when action is list."
            ),
        ),
    ] = DEFAULT_MIN_WAIT_SEC * 1000
    wait_for: Annotated[
        WaitFor,
        Field(
            description=(
                "'exit' waits for the process to exit. 'output' also returns on new stdout or stderr "
                "after the minimum wait. Ignored when action is list."
            )
        ),
    ] = "exit"

    @model_validator(mode="after")
    def validate_action_parameters(self) -> ProcessControlParams:
        if self.action != "list" and self.process_id is None:
            raise ValueError(f"{self.action} requires process_id")
        return self


@dataclass
class _ProcessState:
    process_id: str
    directory: str
    session_name: str
    cmd: str
    started_at: float
    output_offset: int = 0
    spill_seq: int = 0

    @property
    def session_target(self) -> str:
        """tmux target with a leading '=' so the match is exact, not a prefix."""
        return f"={self.session_name}"


@dataclass(frozen=True)
class _ProcessSnapshot:
    output_size: int
    status: ProcessStatus
    exit_code: int | None


@dataclass(frozen=True)
class _StreamDelta:
    head: bytes
    tail: bytes | None
    omitted_bytes: int

    @property
    def truncated(self) -> bool:
        return self.tail is not None


@dataclass(frozen=True)
class _ProcessReport:
    snapshot: _ProcessSnapshot
    output: _StreamDelta
    spill_path: str | None
    original_bytes: int
    saved_bytes: int


@dataclass(frozen=True)
class _WaitOutcome:
    reason: ReturnReason
    waited_sec: float


class BackgroundCodeExecToolProvider(CodeExecToolProvider):
    """Wrap a sandboxed code execution backend with backgroundable ``code_exec`` and ``process_control`` tools.

    Example:
        from stirrup.tools.code_backends.background import BackgroundCodeExecToolProvider
        from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

        agent = Agent(
            client=client,
            name="coder",
            tools=[BackgroundCodeExecToolProvider(DockerCodeExecToolProvider.from_image("my-image-with-tmux"))],
        )

    File operations, output saving, and input uploads delegate to the wrapped backend. A subagent that
    shares this exec env gets the backend's blocking ``code_exec`` tool, without ``process_control``.
    """

    def __init__(
        self,
        backend: CodeExecToolProvider,
        *,
        max_yield_time_sec: float = DEFAULT_MAX_YIELD_TIME_SEC,
        min_wait_sec: float = DEFAULT_MIN_WAIT_SEC,
        output_spill_threshold_chars: int = DEFAULT_OUTPUT_SPILL_THRESHOLD_CHARS,
        output_spill_max_bytes: int = DEFAULT_OUTPUT_SPILL_MAX_BYTES,
    ) -> None:
        """Initialize the provider.

        Args:
            backend: Sandboxed provider to run in. It must run commands through Bash, so it cannot
                     have an ``allowed_commands`` allowlist, and it cannot be
                     ``LocalCodeExecToolProvider``, which rejects the absolute paths used here.
            max_yield_time_sec: Longest single wait a tool call may request.
            min_wait_sec: Shortest wait for a call on a running process. Caps the turns a model spends on
                          rapid polling; a process that exits sooner still returns at once.
            output_spill_threshold_chars: Output bytes shown inline; longer output shows its head and tail.
            output_spill_max_bytes: Size of each process's output ring buffer. Older output is overwritten.
        """
        if isinstance(backend, LocalCodeExecToolProvider):
            raise TypeError("BackgroundCodeExecToolProvider needs a sandboxed backend, not LocalCodeExecToolProvider")
        if backend._allowed_commands is not None:  # noqa: SLF001
            raise ValueError("BackgroundCodeExecToolProvider needs a backend without allowed_commands")
        if max_yield_time_sec < 0:
            raise ValueError("max_yield_time_sec must be non-negative")
        if min_wait_sec < 0:
            raise ValueError("min_wait_sec must be non-negative")
        if output_spill_threshold_chars < 1:
            raise ValueError("output_spill_threshold_chars must be positive")
        if output_spill_max_bytes < 1:
            raise ValueError("output_spill_max_bytes must be positive")
        super().__init__(shell_timeout=backend._shell_timeout)  # noqa: SLF001
        self._backend = backend
        self._max_yield_time_sec = max_yield_time_sec
        self._min_wait_sec = min(min_wait_sec, max_yield_time_sec)
        self._output_spill_threshold_chars = output_spill_threshold_chars
        self._output_spill_max_bytes = output_spill_max_bytes
        # Per instance, so teardown never touches processes another provider started in the same sandbox.
        self._process_root = f"{_PROCESS_ROOT_BASE}/{secrets.token_hex(_PROCESS_ID_BYTES)}"
        self._processes: dict[str, _ProcessState] = {}
        self._cgroup_parent: str | None = None

    @property
    def backend(self) -> CodeExecToolProvider:
        """The wrapped code execution backend."""
        return self._backend

    @property
    def temp_dir(self) -> Path | None:
        return self._backend.temp_dir

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        return await self._backend.run_command(cmd, timeout=timeout)

    async def read_file_bytes(self, path: str) -> bytes:
        return await self._backend.read_file_bytes(path)

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        await self._backend.write_file_bytes(path, content)

    async def file_exists(self, path: str) -> bool:
        return await self._backend.file_exists(path)

    async def is_directory(self, path: str) -> bool:
        return await self._backend.is_directory(path)

    async def list_files(self, path: str) -> list[str]:
        return await self._backend.list_files(path)

    async def view_image(self, path: str) -> ImageContentBlock:
        return await self._backend.view_image(path)

    def output_source_roots(self) -> tuple[OutputSourceRoot, ...]:
        return self._backend.output_source_roots()

    async def resolve_output_source(self, source_path: str) -> str:
        return await self._backend.resolve_output_source(source_path)

    async def output_destination_identity(self, destination: str, output_root: Path | str) -> str:
        return await self._backend.output_destination_identity(destination, output_root)

    async def save_output_files(
        self,
        paths: list[str],
        output_dir: Path | str,
        dest_env: CodeExecToolProvider | None = None,
    ) -> SaveOutputFilesResult:
        return await self._backend.save_output_files(paths, output_dir, dest_env)

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: CodeExecToolProvider | None = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        return await self._backend.upload_files(*paths, source_env=source_env, dest_dir=dest_dir)

    async def _exec(self, command: str, *, timeout_sec: int) -> CommandResult:
        """Run a command, writing it to a script in chunks when it is too long for one argument."""
        if _fits_inline(command):
            return await self._backend.run_command(command, timeout=timeout_sec)
        script = shlex.quote(f"{self._process_root}/commands/{secrets.token_hex(_PROCESS_ID_BYTES)}.sh")
        encoded = base64.b64encode(command.encode()).decode()
        chunks = [encoded[i : i + _COMMAND_CHUNK_CHARS] for i in range(0, len(encoded), _COMMAND_CHUNK_CHARS)]
        create = f"mkdir -p {shlex.quote(self._process_root + '/commands')} && : > {script}"
        for step in [create, *(f"printf %s {shlex.quote(chunk)} | base64 -d >> {script}" for chunk in chunks)]:
            written = await self._backend.run_command(step, timeout=_ENVIRONMENT_PROBE_TIMEOUT_SEC)
            if written.exit_code != 0:
                raise RuntimeError(f"Could not write command script: {written.stderr or written.stdout or 'no output'}")
        return await self._backend.run_command(f"trap 'rm -f {script}' EXIT; . {script}", timeout=timeout_sec)

    async def _raise_if_environment_unavailable(self) -> None:
        probe = await self._backend.run_command(_ENVIRONMENT_PROBE_COMMAND, timeout=_ENVIRONMENT_PROBE_TIMEOUT_SEC)
        if probe.exit_code == 0:
            return
        raise RuntimeError(f"Code execution environment is unavailable: {probe.stderr or probe.stdout or 'no output'}")

    async def _ensure_background_tools(self) -> None:
        probe = await self._backend.run_command(_PROBE_BACKGROUND_TOOLS_COMMAND, timeout=_ENVIRONMENT_PROBE_TIMEOUT_SEC)
        if probe.exit_code != 0:
            detail = (probe.stdout or probe.stderr or "no output").strip()
            raise RuntimeError(
                "BackgroundCodeExecToolProvider requires " + ", ".join(BACKGROUND_TOOLS) + f" in the sandbox: {detail}"
            )
        # Installed is not enough: tmux needs a pty, which a non-root user may not get.
        session = shlex.quote(f"{_SESSION_PREFIX}probe_{secrets.token_hex(_PROCESS_ID_BYTES)}")
        tmux = await self._backend.run_command(
            f"tmux new-session -d -s {session} true", timeout=_ENVIRONMENT_PROBE_TIMEOUT_SEC
        )
        if tmux.exit_code != 0:
            detail = (tmux.stderr or tmux.stdout or "no output").strip()
            raise RuntimeError(f"tmux cannot start a session: {detail}")

    async def _control_exec(self, command: str) -> CommandResult:
        try:
            with anyio.fail_after(_CONTROL_TIMEOUT_SEC + _ENVIRONMENT_PROBE_TIMEOUT_SEC):
                result = await self._exec(command, timeout_sec=_CONTROL_TIMEOUT_SEC)
        except TimeoutError:
            raise RuntimeError(f"Process control timed out after {_CONTROL_TIMEOUT_SEC} seconds") from None
        if result.error_kind == "timeout":
            raise RuntimeError(f"Process control timed out after {_CONTROL_TIMEOUT_SEC} seconds")
        return result

    async def __aenter__(self) -> list[Tool]:  # ty: ignore[invalid-method-override]
        await self._backend.__aenter__()
        try:
            await self._ensure_background_tools()
            root = shlex.quote(self._process_root)
            probe_file = shlex.quote(f"{self._process_root}/.rwprobe")
            setup = await self._control_exec(
                f'mkdir -p {root} && printf ok > {probe_file} && [ "$(cat {probe_file})" = ok ] && rm -f {probe_file}'
            )
            if setup.exit_code != 0:
                raise RuntimeError(
                    f"{self._process_root} must be writable and readable: {setup.stderr or setup.stdout or 'no output'}"
                )
            self._cgroup_parent = await self._probe_cgroup_parent()
        except BaseException as error:
            await self._backend.__aexit__(type(error), error, error.__traceback__)
            raise
        return [self._code_exec_tool(), self._process_control_tool()]

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        # Unconditional: a start whose response was lost leaves an untracked process.
        try:
            with anyio.CancelScope(shield=True):
                result = await self._control_exec(self._kill_all_command())
            if result.exit_code != 0:
                _logger.error("Failed to clean up all background processes: %s", result.stderr or result.stdout)
        except Exception:
            _logger.exception("Failed to clean up background processes during teardown")
        finally:
            self._processes.clear()
            await self._backend.__aexit__(exc_type, exc_val, exc_tb)

    def _capped_yield_sec(self, yield_time_ms: float) -> tuple[float, bool]:
        # No schema minimum: Anthropic rejects it, so negative yields clamp to zero here.
        requested = max(yield_time_ms, 0) / 1000
        return min(requested, self._max_yield_time_sec), requested > self._max_yield_time_sec

    def _max_yield_time_ms(self) -> float:
        return self._max_yield_time_sec * 1000

    def _code_exec_tool(
        self,
    ) -> Tool[BackgroundableCodeExecutionParams, ToolUseCountMetadata]:
        async def executor(
            params: BackgroundableCodeExecutionParams,
        ) -> ToolResult[ToolUseCountMetadata]:
            yield_time_sec, capped = self._capped_yield_sec(params.yield_time_ms)
            if not params.cmd:
                waited_sec = max(yield_time_sec, self._min_wait_sec)
                if waited_sec > 0:
                    await anyio.sleep(waited_sec)
                fields = [f"<waited_ms>{_ms(waited_sec)}</waited_ms>"]
                if capped:
                    fields.append(f"<yield_capped_to>{self._max_yield_time_ms():g}</yield_capped_to>")
                if waited_sec > yield_time_sec:
                    fields.append(f"<yield_raised_to>{_ms(self._min_wait_sec)}</yield_raised_to>")
                return ToolResult(
                    content="<process>" + "".join(fields) + "</process>",
                    metadata=ToolUseCountMetadata(),
                )
            try:
                process = await self._start(params.cmd, params.workdir)
                outcome = await self._poll_until(
                    process, yield_time_sec, wait_for=params.wait_for, min_sec=self._min_wait_sec
                )
                return ToolResult(
                    content=await self._result(process, outcome, capped, requested_sec=yield_time_sec),
                    metadata=ToolUseCountMetadata(),
                )
            except ProcessOperationError as error:
                return await self._failure(error)

        return Tool(
            name="code_exec",
            description=(
                "Start a Bash command without a TTY and wait yield_time_ms, clamped between "
                f"{_ms(self._min_wait_sec)} ms and {_ms(self._max_yield_time_sec)} ms. Returns immediately if the command exits. "
                "Returns its ID, status, timing, combined stdout and stderr "
                "produced so far, and an exit code when available. Running commands continue until they exit or "
                "process_control terminates them. "
                f"Non-empty output includes an inline preview of up to "
                f"{self._output_spill_threshold_chars:,} output bytes and a saved file. Interactive stdin is "
                "unavailable; use a pipe, heredoc, or file for input. code_exec keeps running commands alive after "
                "it returns. Run long-lived services directly and poll them. Do not use '&'; child processes are "
                "stopped when the command exits. An empty cmd only waits and returns no process ID."
            ),
            parameters=BackgroundableCodeExecutionParams,
            executor=executor,
        )

    def _process_control_tool(
        self,
    ) -> Tool[ProcessControlParams, ToolUseCountMetadata]:
        async def executor(
            params: ProcessControlParams,
        ) -> ToolResult[ToolUseCountMetadata]:
            try:
                if params.action == "list":
                    return ToolResult(content=await self._list(), metadata=ToolUseCountMetadata())
                process = self._processes.get(params.process_id or "")
                if process is None:
                    return ToolResult(
                        content=f"Unknown process: {params.process_id}",
                        success=False,
                        metadata=ToolUseCountMetadata(),
                    )
                yield_time_sec, capped = self._capped_yield_sec(params.yield_time_ms)
                match params.action:
                    case "poll":
                        pass
                    case "terminate":
                        await self._terminate(process)
                    case _:
                        assert_never(params.action)
                min_sec = self._min_wait_sec if params.action == "poll" else 0
                outcome = await self._poll_until(process, yield_time_sec, wait_for=params.wait_for, min_sec=min_sec)
                requested_sec = yield_time_sec if params.action == "poll" else None
                return ToolResult(
                    content=await self._result(process, outcome, capped, requested_sec=requested_sec),
                    metadata=ToolUseCountMetadata(),
                )
            except ProcessOperationError as error:
                return await self._failure(error)

        return Tool(
            name="process_control",
            description=(
                "Poll, terminate, or list processes started by code_exec. poll waits yield_time_ms, clamped between "
                f"{_ms(self._min_wait_sec)} ms and {_ms(self._max_yield_time_sec)} ms. Returns immediately if the process exits. "
                "terminate stops the command and its process group. list returns only running processes; "
                "completed or terminated processes remain pollable by ID. Polling a finished process "
                "returns immediately with its last status and any unread output. Poll and terminate return "
                "only new output in code_exec's preview and saved-file format."
            ),
            parameters=ProcessControlParams,
            executor=executor,
        )

    async def _probe_cgroup_parent(self) -> str | None:
        result = await self._backend.run_command(_CGROUP_PROBE_COMMAND, timeout=_ENVIRONMENT_PROBE_TIMEOUT_SEC)
        parent = (result.stdout or "").strip().splitlines()
        if result.exit_code != 0 or not parent or not parent[-1].startswith(_CGROUP_FS):
            return None
        return parent[-1]

    async def _failure(self, error: ProcessOperationError) -> ToolResult[ToolUseCountMetadata]:
        await self._raise_if_environment_unavailable()
        return ToolResult(content=str(error), success=False, metadata=ToolUseCountMetadata())

    async def _start(self, cmd: str, workdir: str | None) -> _ProcessState:
        process_id = f"{_PROCESS_ID_PREFIX}{secrets.token_hex(_PROCESS_ID_BYTES)}"
        directory = f"{self._process_root}/{process_id}"
        session_name = f"{_SESSION_PREFIX}{process_id}"
        quoted_directory = shlex.quote(directory)
        shell_command = f"bash {shlex.quote(directory + '/command.sh')}"
        if workdir is not None:
            shell_command = f"{_cd_command(workdir)} && {shell_command}"
        output_log = shlex.quote(directory + "/output.log")
        ring_path = shlex.quote(directory + "/ring.sh")
        written_path = shlex.quote(directory + "/written")
        command_pgid = shlex.quote(directory + "/command.pgid")
        command_exit = shlex.quote(directory + "/command_exit")
        command_exit_tmp = shlex.quote(directory + "/command_exit.tmp")
        cgroup_file = shlex.quote(directory + "/cgroup")
        command_body = (
            f"printf '%s\\n' \"$$\" > {command_pgid}; "
            f"if test -s {cgroup_file}; then cg=$(cat {cgroup_file}); "
            f'printf \'%s\\n\' "$$" > "$cg/cgroup.procs" || true; fi; '
            f"{shell_command}; "
            f"printf '%s\\n' \"$?\" > {command_exit_tmp} && mv {command_exit_tmp} {command_exit}"
        )
        runner = (
            f"printf '%s\\n' \"$$\" > {shlex.quote(directory + '/runner.pid')}\n"
            f"printf 0 > {written_path}\n"
            f"setsid bash -c {shlex.quote(command_body)} < /dev/null 2>&1 "
            f"| bash {ring_path} {output_log} {written_path} {self._output_spill_max_bytes} &\n"
            "ring=$!\n"
            f"while ! test -f {command_exit}; do sleep 0.05; done\n"
            f"{self._cgroup_kill_command(quoted_directory)}\n"
            f"{self._signal_pgid_command(quoted_directory, 'TERM')}\n"
            f"{self._signal_pgid_command(quoted_directory, 'KILL')}\n"
            'wait "$ring" || true\n'
            f"code=$(cat {command_exit})\n"
            f"printf '%s\\n' \"$code\" > {shlex.quote(directory + '/exit_code.tmp')} "
            f"&& mv {shlex.quote(directory + '/exit_code.tmp')} {shlex.quote(directory + '/exit_code')}\n"
        )
        command_text = base64.b64encode(cmd.encode()).decode()
        runner_text = base64.b64encode(runner.encode()).decode()
        ring_text = base64.b64encode(_RING_CAPTURE_SCRIPT.encode()).decode()
        cgroup_setup = ""
        if self._cgroup_parent is not None:
            cgroup = shlex.quote(f"{self._cgroup_parent}/{process_id}")
            cgroup_setup = f"if mkdir {cgroup} 2>/dev/null; then printf '%s\\n' {cgroup} > {cgroup_file}; fi && "
        setup = (
            f"mkdir -p {quoted_directory} && "
            f"{cgroup_setup}"
            f"touch {output_log} && "
            f"printf %s {shlex.quote(command_text)} | base64 -d > {shlex.quote(directory + '/command.sh')} && "
            f"printf %s {shlex.quote(ring_text)} | base64 -d > {ring_path} && "
            f"printf %s {shlex.quote(runner_text)} | base64 -d > {shlex.quote(directory + '/run.sh')} && "
            f"tmux new-session -d -s {shlex.quote(session_name)} bash {shlex.quote(directory + '/run.sh')} "
            f"|| {{ rm -rf {quoted_directory}; exit 1; }}"
        )
        result = await self._control_exec(setup)
        if result.exit_code != 0:
            raise ProcessOperationError(f"Failed to start process: {result.stderr or result.stdout or 'no output'}")
        process = _ProcessState(process_id, directory, session_name, cmd, time.monotonic())
        self._processes[process_id] = process
        return process

    async def _poll_until(
        self, process: _ProcessState, seconds: float, *, wait_for: WaitFor, min_sec: float = 0
    ) -> _WaitOutcome:
        """Wait for exit, or output once ``min_sec`` has passed, up to ``max(seconds, min_sec)``."""
        started = time.monotonic()
        deadline = started + max(seconds, min_sec)
        interval = _PROCESS_POLL_INITIAL_SEC
        reason: ReturnReason = "no_wait"
        while time.monotonic() < deadline:
            snapshot = await self._snapshot(process)
            if snapshot.status != "running":
                reason = "exit"
                break
            if (
                wait_for == "output"
                and snapshot.output_size > process.output_offset
                and time.monotonic() - started >= min_sec
            ):
                reason = "output"
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                reason = "yield_expired"
                break
            await anyio.sleep(min(interval, remaining))
            interval = min(interval * 2, _PROCESS_POLL_MAX_SEC)
            reason = "yield_expired"
        return _WaitOutcome(reason=reason, waited_sec=time.monotonic() - started)

    @staticmethod
    def _collect_tree_command(directory: str) -> str:
        return (
            f"if test -f {directory}/runner.pid; then root=$(cat {directory}/runner.pid); "
            'collect_tree() { for child in $(pgrep -P "$1" 2>/dev/null || true); '
            'do collect_tree "$child"; done; printf \'%s\\n\' "$1"; }; '
            f'collect_tree "$root" > {directory}/termination_targets; fi'
        )

    @staticmethod
    def _signal_tree_command(directory: str, signal: Literal["TERM", "KILL"]) -> str:
        return (
            f"if test -f {directory}/termination_targets; then while IFS= read -r pid; "
            f'do kill -{signal} "$pid" 2>/dev/null || true; done < {directory}/termination_targets; fi'
        )

    @staticmethod
    def _cgroup_kill_command(directory: str) -> str:
        return (
            f"if test -s {directory}/cgroup; then cg=$(cat {directory}/cgroup); "
            'if test -w "$cg/cgroup.kill"; then printf 1 > "$cg/cgroup.kill"; fi; fi'
        )

    @staticmethod
    def _cgroup_rmdir_command(directory: str) -> str:
        return f'if test -s {directory}/cgroup; then rmdir "$(cat {directory}/cgroup)" 2>/dev/null || true; fi'

    @staticmethod
    def _verify_cgroup_stopped_command(directory: str) -> str:
        return (
            f"if test -s {directory}/cgroup; then cg=$(cat {directory}/cgroup); "
            'left=$(tr -d " \\n" < "$cg/cgroup.procs" 2>/dev/null || true); '
            'if test -n "$left"; then printf "Failed to terminate cgroup %s:%s\\n" "$cg" "$left" >&2; false; fi; fi'
        )

    @staticmethod
    def _signal_pgid_command(directory: str, signal: Literal["TERM", "KILL"]) -> str:
        return (
            f"if test -f {directory}/command.pgid; then pgid=$(cat {directory}/command.pgid); "
            'case "$pgid" in ""|*[!0-9]*|0|1) ;; *) '
            f'kill -{signal} -- "-$pgid" 2>/dev/null || true ;; esac; fi'
        )

    @staticmethod
    def _verify_tree_stopped_command(directory: str) -> str:
        return (
            f"survivors=''; if test -f {directory}/termination_targets; then while IFS= read -r pid; do "
            'state=$(ps -o stat= -p "$pid" 2>/dev/null | tr -d " "); '
            'case "$state" in ""|Z*) ;; *) survivors="$survivors $pid" ;; esac; '
            f"done < {directory}/termination_targets; fi; "
            'if test -n "$survivors"; then printf "Failed to terminate process tree:%s\\n" "$survivors" >&2; false; fi'
        )

    @staticmethod
    def _verify_pgid_stopped_command(directory: str) -> str:
        return (
            f"if test -f {directory}/command.pgid; then pgid=$(cat {directory}/command.pgid); "
            'case "$pgid" in ""|*[!0-9]*|0|1) ;; *) '
            'left=$(pgrep -g "$pgid" 2>/dev/null || true); '
            'if test -n "$left"; then printf "Failed to terminate process group %s:%s\\n" "$pgid" "$left" >&2; false; fi ;; esac; fi'
        )

    @classmethod
    def _prepare_termination_command(cls, directory: str, session_target: str) -> str:
        return (
            f"{cls._cgroup_kill_command(directory)}; "
            f"{cls._signal_pgid_command(directory, 'TERM')}; "
            f"if ! test -f {directory}/exit_code; then "
            f"{cls._collect_tree_command(directory)}; "
            f"{cls._signal_tree_command(directory, 'TERM')}; "
            f"tmux kill-session -t {session_target} >/dev/null 2>&1 || true; "
            "fi"
        )

    @classmethod
    def _finish_termination_command(cls, directory: str) -> str:
        return (
            f"{cls._cgroup_kill_command(directory)}; "
            f"{cls._signal_pgid_command(directory, 'KILL')}; "
            f"if ! test -f {directory}/exit_code; then {cls._signal_tree_command(directory, 'KILL')}; fi; "
            f"{cls._verify_cgroup_stopped_command(directory)} && "
            f"{cls._verify_pgid_stopped_command(directory)} && "
            f"{cls._verify_tree_stopped_command(directory)} && "
            f"{cls._cgroup_rmdir_command(directory)} && "
            f"if ! test -f {directory}/exit_code; then touch {directory}/terminated; fi"
        )

    @classmethod
    def _kill_tree_command(cls, process: _ProcessState) -> str:
        directory = shlex.quote(process.directory)
        prepare = cls._prepare_termination_command(directory, shlex.quote(process.session_target))
        return (
            f"status=0; {prepare} || status=$?; "
            f'sleep {_TERMINATE_GRACE_SEC}; {cls._finish_termination_command(directory)} || status=$?; exit "$status"'
        )

    def _kill_all_command(self) -> str:
        """Terminate every process under the process root in a command whose size is fixed."""
        cls = type(self)
        each = f'for dir in {shlex.quote(self._process_root)}/{_PROCESS_ID_PREFIX}*; do test -d "$dir" || continue; '
        directory = '"$dir"'
        session_target = f'"={_SESSION_PREFIX}${{dir##*/}}"'
        return (
            f"status=0; found=; {each}found=1; "
            f"{cls._prepare_termination_command(directory, session_target)} || status=$?; done; "
            f'if test -n "$found"; then sleep {_TERMINATE_GRACE_SEC}; fi; '
            f"{each}{cls._finish_termination_command(directory)} || status=$?; done; "
            f'rm -rf {shlex.quote(self._process_root)}; exit "$status"'
        )

    async def _terminate(self, process: _ProcessState) -> None:
        if (await self._snapshot(process)).status != "running":
            return
        result = await self._control_exec(self._kill_tree_command(process))
        if result.exit_code != 0:
            error = result.stderr or result.stdout or "no output"
            raise ProcessOperationError(f"Failed to terminate process ({result.exit_code}): {error}")

    @staticmethod
    def _state_script(process: _ProcessState) -> str:
        """Print one line: bytes written, status, exit code."""
        directory = shlex.quote(process.directory)
        target = shlex.quote(process.session_target)
        return (
            f"if test -f {directory}/exit_code; then state=completed; code=$(cat {directory}/exit_code); "
            f"elif test -f {directory}/terminated; then state=terminated; code=; "
            f"elif tmux has-session -t {target} 2>/dev/null; then state=running; code=; "
            f"elif test -f {directory}/exit_code; then state=completed; code=$(cat {directory}/exit_code); "
            "else state=terminated; code=; fi; "
            f"out=$(cat {directory}/written 2>/dev/null || printf 0); out=$(( ${{out:-0}} )); "
            'printf \'%s %s %s\\n\' "$out" "$state" "$code"; '
        )

    def _spill_script(self, process: _ProcessState, spill_path: str) -> str:
        """Copy the unread ring window to spill_path and print path/sizes plus a 30k head/tail."""
        limit = self._output_spill_threshold_chars
        half = max(limit // 2, 1)
        max_ring = self._output_spill_max_bytes
        output_log = shlex.quote(process.directory + "/output.log")
        spill = shlex.quote(spill_path)
        offset = process.output_offset
        encode = "base64 | tr -d '\\n'"
        return (
            f'delta=$(( out - {offset} )); if [ "$delta" -lt 0 ]; then delta=0; fi; '
            f"if [ \"$delta\" -eq 0 ]; then printf '\\n\\n\\n'; "
            f"else "
            f'dropped=0; if [ "$delta" -gt {max_ring} ]; then dropped=$(( delta - {max_ring} )); fi; '
            f"length=$(( delta - dropped )); "
            f"start=$(( ( {offset} + dropped ) % {max_ring} )); "
            f"size=$(wc -c < {output_log} 2>/dev/null || printf 0); size=$(( ${{size:-0}} )); "
            f'if [ $(( start + length )) -le "$size" ]; then '
            f'tail -c +$(( start + 1 )) {output_log} | head -c "$length" > {spill}; '
            f"else "
            f"first=$(( size - start )); "
            f'tail -c +$(( start + 1 )) {output_log} | head -c "$first" > {spill}; '
            f"head -c $(( length - first )) {output_log} >> {spill}; "
            f"fi; "
            f"spill=$(wc -c < {spill}); "
            f'printf \'%s %s %s\\n\' {spill} "$delta" "$spill"; '
            f'if [ "$spill" -le {limit} ]; then '
            f"cat {spill} | {encode}; printf '\\n\\n'; "
            f"else "
            f"head -c {half} {spill} | {encode}; printf '\\n'; "
            f"tail -c {half} {spill} | {encode}; printf '\\n'; "
            f"fi; fi; "
        )

    @staticmethod
    def _parse_state_line(line: str) -> _ProcessSnapshot:
        parts = line.split()
        if len(parts) < 2:
            raise ProcessOperationError("Failed to parse process state")
        try:
            output_size = int(parts[0])
            exit_code = int(parts[2]) if len(parts) > 2 else None
        except ValueError as error:
            raise ProcessOperationError(f"Failed to parse process state: {line!r}") from error
        status: ProcessStatus
        match parts[1]:
            case "running":
                status = "running"
            case "completed":
                status = "completed"
            case "terminated":
                status = "terminated"
            case other:
                raise ProcessOperationError(f"Failed to parse process status: {other!r}")
        return _ProcessSnapshot(output_size=output_size, status=status, exit_code=exit_code)

    async def _check_exec(self, script: str, action: str) -> str:
        """Run a read-only check, retrying only when the exec did not run the script to its end."""
        start_line = f"{_CHECK_START_MARKER}\n"
        marker_line = f"{_CHECK_COMPLETE_MARKER}\n"
        attempts = len(_CHECK_RETRY_DELAYS_SEC) + 1
        detail = ""
        for attempt in range(attempts):
            result = await self._control_exec(
                f"printf '%s\\n' {_CHECK_START_MARKER}; {script}printf '%s\\n' {_CHECK_COMPLETE_MARKER}"
            )
            stdout = result.stdout or ""
            if stdout.endswith(marker_line) and start_line in stdout:
                # Some backends merge their own stderr into the output, so take what follows the start marker.
                return stdout.removesuffix(marker_line).rpartition(start_line)[2]
            detail = f"exit code {result.exit_code}: {result.stderr or stdout or 'no output'}"
            if attempt < len(_CHECK_RETRY_DELAYS_SEC):
                delay = _CHECK_RETRY_DELAYS_SEC[attempt]
                _logger.warning("Process check to %s did not complete (%s); retrying in %ss", action, detail, delay)
                await anyio.sleep(delay)
        await self._raise_if_environment_unavailable()
        raise RuntimeError(f"Failed to {action}: the check did not complete after {attempts} attempts ({detail})")

    async def _snapshot(self, process: _ProcessState) -> _ProcessSnapshot:
        return self._parse_state_line(await self._check_exec(self._state_script(process), "inspect process"))

    async def _report(self, process: _ProcessState) -> _ProcessReport:
        """Read state, spill the unread delta, and return a 30k head/tail."""
        next_output = f"{process.directory}/output_{process.spill_seq + 1}.log"
        script = self._state_script(process) + self._spill_script(process, next_output)
        stdout = await self._check_exec(script, "read process output")
        lines = stdout.split("\n")
        if len(lines) < 4:
            raise ProcessOperationError(f"Failed to parse process output: {stdout!r}")
        snapshot = self._parse_state_line(lines[0])
        if not lines[1].strip():
            process.output_offset = snapshot.output_size
            return _ProcessReport(
                snapshot=snapshot,
                output=_StreamDelta(head=b"", tail=None, omitted_bytes=0),
                spill_path=None,
                original_bytes=0,
                saved_bytes=0,
            )
        meta = lines[1].split()
        if len(meta) != 3:
            raise ProcessOperationError(f"Failed to parse process spill: {stdout!r}")
        try:
            original_bytes = int(meta[1])
            spill_size = int(meta[2])
        except ValueError as error:
            raise ProcessOperationError(f"Failed to parse process spill: {stdout!r}") from error
        output = self._parse_delta(lines[2], lines[3], spill_size)
        process.output_offset = snapshot.output_size
        process.spill_seq += 1
        return _ProcessReport(
            snapshot=snapshot,
            output=output,
            spill_path=meta[0],
            original_bytes=original_bytes,
            saved_bytes=spill_size,
        )

    @staticmethod
    def _parse_delta(head_line: str, tail_line: str, delta_size: int) -> _StreamDelta:
        try:
            head = base64.b64decode(head_line, validate=True)
            tail = base64.b64decode(tail_line, validate=True) if tail_line else None
        except binascii.Error as error:
            raise ProcessOperationError("Failed to decode process output") from error
        omitted = max(delta_size - len(head) - len(tail or b""), 0) if tail is not None else 0
        return _StreamDelta(head=head, tail=tail, omitted_bytes=omitted)

    async def _result(
        self, process: _ProcessState, outcome: _WaitOutcome, capped: bool, *, requested_sec: float | None = None
    ) -> str:
        report = await self._report(process)
        snapshot = report.snapshot
        fields = [
            f"<process_id>{process.process_id}</process_id>",
            f"<status>{snapshot.status}</status>",
        ]
        if snapshot.exit_code is not None:
            fields.append(f"<exit_code>{snapshot.exit_code}</exit_code>")
        fields.append(f"<elapsed_ms>{_ms(time.monotonic() - process.started_at)}</elapsed_ms>")
        fields.append(f"<waited_ms>{_ms(outcome.waited_sec)}</waited_ms>")
        # The process can exit between the wait's last check and this report.
        reason = "exit" if snapshot.status != "running" else outcome.reason
        fields.append(f"<returned_because>{reason}</returned_because>")
        if capped:
            fields.append(f"<yield_capped_to>{self._max_yield_time_ms():g}</yield_capped_to>")
        # Report the floor only when it held the call: past the requested wait and not ended by exit.
        if (
            requested_sec is not None
            and requested_sec < self._min_wait_sec
            and outcome.waited_sec > requested_sec
            and reason != "exit"
        ):
            fields.append(f"<yield_raised_to>{_ms(self._min_wait_sec):g}</yield_raised_to>")
        fields.extend(self._output_fields(report))
        return "<process>\n  " + "\n  ".join(fields) + "\n</process>"

    async def _list(self) -> str:
        processes = list(self._processes.values())
        if not processes:
            return '<processes count="0"/>'
        stdout = await self._check_exec("".join(self._state_script(process) for process in processes), "list processes")
        lines = stdout.splitlines()
        if len(lines) != len(processes):
            raise ProcessOperationError(f"Failed to parse process list: {stdout!r}")
        entries = []
        for process, line in zip(processes, lines, strict=True):
            snapshot = self._parse_state_line(line)
            preview = (
                process.cmd
                if len(process.cmd) <= _LIST_COMMAND_PREVIEW_CHARS
                else (process.cmd[:_LIST_COMMAND_PREVIEW_CHARS] + "...")
            )
            fields = [f"<process_id>{process.process_id}</process_id>", f"<status>{snapshot.status}</status>"]
            if snapshot.exit_code is not None:
                fields.append(f"<exit_code>{snapshot.exit_code}</exit_code>")
            fields.append(f"<elapsed_ms>{_ms(time.monotonic() - process.started_at)}</elapsed_ms>")
            fields.append(f"<cmd>{preview}</cmd>")
            if snapshot.status == "running":
                entries.append("  <process>" + "".join(fields) + "</process>")
        if not entries:
            return '<processes count="0"/>'
        return f'<processes count="{len(entries)}">\n' + "\n".join(entries) + "\n</processes>"

    def _output_fields(self, report: _ProcessReport) -> list[str]:
        if report.original_bytes == 0 or report.spill_path is None:
            return []
        # Byte windows can split a multibyte character; replace never raises.
        head = report.output.head.decode(errors="replace")
        dropped = max(0, report.original_bytes - self._output_spill_max_bytes)
        if report.output.truncated or dropped:
            tail = (report.output.tail or b"").decode(errors="replace")
            body = (
                f"{head}\n...[{report.output.omitted_bytes} bytes omitted]...\n{tail}"
                if report.output.truncated
                else head
            )
        else:
            body = head
        file_attributes = [
            f"path={quoteattr(report.spill_path)}",
            f"bytes={quoteattr(str(report.saved_bytes))}",
        ]
        if dropped:
            file_attributes.append(
                "note="
                + quoteattr(
                    f"Only the latest {_format_output_size(self._output_spill_max_bytes)} was preserved; "
                    f"{dropped:,} earlier bytes were overwritten."
                )
            )
        file = f"<file {' '.join(file_attributes)}/>"
        # Bytes stay unescaped, like the blocking code_exec, because the model copies them into edits.
        return [
            "<output>",
            f'  <raw preview="true">{body}</raw>',
            f"  {file}",
            "</output>",
        ]
