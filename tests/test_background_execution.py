"""Tests for BackgroundCodeExecToolProvider against a Bash backend on the host."""

import inspect
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

import anyio
import pytest

from stirrup.core.models import ImageContentBlock, Tool, ToolResult
from stirrup.tools.code_backends.background import (
    BACKGROUND_TOOLS,
    BackgroundableCodeExecutionParams,
    BackgroundCodeExecToolProvider,
    ProcessControlParams,
)
from stirrup.tools.code_backends.base import CodeExecToolProvider, CommandResult
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or any(shutil.which(tool) is None for tool in BACKGROUND_TOOLS),
    reason="needs Linux with " + ", ".join(BACKGROUND_TOOLS),
)


class HostBashProvider(CodeExecToolProvider):
    """Unsandboxed Bash on the host, standing in for a container backend."""

    def __init__(self, allowed_commands: list[str] | None = None) -> None:
        super().__init__(allowed_commands=allowed_commands)
        self._dir: Path | None = None
        self.exited = False

    async def __aenter__(self) -> Tool:
        self._dir = Path(tempfile.mkdtemp(prefix="stirrup_bg_test_"))
        return self.get_code_exec_tool()

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
        self.exited = True

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        try:
            with anyio.fail_after(timeout or self._shell_timeout):
                result = await anyio.run_process(["bash", "-c", cmd], cwd=self._dir, check=False)
        except TimeoutError:
            return CommandResult(exit_code=1, stdout="", stderr="timed out", error_kind="timeout")
        return CommandResult(
            exit_code=result.returncode,
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
        )

    async def read_file_bytes(self, path: str) -> bytes:
        assert self._dir is not None
        return (self._dir / path).read_bytes()

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        assert self._dir is not None
        (self._dir / path).write_bytes(content)

    async def file_exists(self, path: str) -> bool:
        assert self._dir is not None
        return (self._dir / path).is_file()

    async def is_directory(self, path: str) -> bool:
        assert self._dir is not None
        return (self._dir / path).is_dir()

    async def list_files(self, path: str) -> list[str]:
        assert self._dir is not None
        return [str(p.relative_to(self._dir / path)) for p in (self._dir / path).rglob("*") if p.is_file()]

    async def view_image(self, path: str) -> ImageContentBlock:
        raise NotImplementedError


def _tools(tools: list[Tool]) -> tuple[Tool, Tool]:
    by_name = {tool.name: tool for tool in tools}
    return by_name["code_exec"], by_name["process_control"]


async def _run(tool: Tool, params: object) -> ToolResult:
    result = tool.executor(params)
    if inspect.isawaitable(result):
        result = await result
    assert isinstance(result, ToolResult)
    return result


async def _start(
    code_exec: Tool,
    cmd: str,
    yield_time_ms: float = 0,
    wait_for: Literal["exit", "output"] = "exit",
    workdir: str | None = None,
) -> str:
    params = BackgroundableCodeExecutionParams(cmd=cmd, yield_time_ms=yield_time_ms, wait_for=wait_for, workdir=workdir)
    result = await _run(code_exec, params)
    assert isinstance(result.content, str)
    return result.content


async def _control(
    process_control: Tool,
    action: Literal["poll", "terminate", "list"],
    process_id: str | None = None,
    yield_time_ms: float = 10_000,
) -> str:
    params = ProcessControlParams(action=action, process_id=process_id, yield_time_ms=yield_time_ms)
    result = await _run(process_control, params)
    assert isinstance(result.content, str)
    return result.content


def _process_id(content: str) -> str:
    return content.split("<process_id>")[1].split("</process_id>")[0]


def _pid_alive(pid: int) -> bool:
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False)
    return state.stdout.strip() not in ("", "Z")


@pytest.fixture
def provider() -> BackgroundCodeExecToolProvider:
    return BackgroundCodeExecToolProvider(HostBashProvider(), min_wait_sec=0, max_yield_time_sec=30)


class TestBackgroundCodeExecToolProvider:
    async def test_command_that_exits_returns_output_and_exit_code(
        self, provider: BackgroundCodeExecToolProvider
    ) -> None:
        async with provider as tools:
            code_exec, _ = _tools(tools)
            content = await _start(code_exec, "echo hello; echo oops >&2; exit 3", yield_time_ms=10_000)
        assert "<status>completed</status>" in content
        assert "<exit_code>3</exit_code>" in content
        assert "<returned_because>exit</returned_because>" in content
        assert "hello" in content
        assert "oops" in content

    async def test_running_command_keeps_running_and_is_pollable(
        self, provider: BackgroundCodeExecToolProvider
    ) -> None:
        async with provider as tools:
            code_exec, process_control = _tools(tools)
            content = await _start(
                code_exec, "echo first; sleep 2; echo second", yield_time_ms=10_000, wait_for="output"
            )
            assert "<status>running</status>" in content
            assert "first" in content
            process_id = _process_id(content)

            listed = await _control(process_control, "list")
            assert process_id in listed

            polled = await _control(process_control, "poll", process_id, yield_time_ms=10_000)
            assert "<status>completed</status>" in polled
            assert "<exit_code>0</exit_code>" in polled
            assert "second" in polled
            assert "first" not in polled

            assert await _control(process_control, "list") == '<processes count="0"/>'

    async def test_wait_for_output_returns_on_new_output(self, provider: BackgroundCodeExecToolProvider) -> None:
        async with provider as tools:
            code_exec, _ = _tools(tools)
            content = await _start(
                code_exec, "sleep 0.5; echo ready; sleep 30", yield_time_ms=20_000, wait_for="output"
            )
        assert "<returned_because>output</returned_because>" in content
        assert "ready" in content

    async def test_terminate_stops_the_process_tree(self, provider: BackgroundCodeExecToolProvider) -> None:
        async with provider as tools:
            code_exec, process_control = _tools(tools)
            content = await _start(code_exec, "bash -c 'sleep 300 & echo child=$!; wait'", yield_time_ms=1_000)
            child = int(content.split("child=")[1].split()[0])
            assert _pid_alive(child)

            terminated = await _control(process_control, "terminate", _process_id(content))
            assert "<status>terminated</status>" in terminated
            assert not _pid_alive(child)

    async def test_exit_terminates_running_processes_and_exits_backend(self) -> None:
        backend = HostBashProvider()
        provider = BackgroundCodeExecToolProvider(backend, min_wait_sec=0)
        async with provider as tools:
            code_exec, _ = _tools(tools)
            content = await _start(code_exec, "echo pid=$$; sleep 300")
            pid = int(content.split("pid=")[1].split()[0])
            assert _pid_alive(pid)
        assert not _pid_alive(pid)
        assert backend.exited

    async def test_large_output_is_previewed_and_saved(self) -> None:
        provider = BackgroundCodeExecToolProvider(HostBashProvider(), min_wait_sec=0, output_spill_threshold_chars=100)
        async with provider as tools:
            code_exec, _ = _tools(tools)
            content = await _start(code_exec, "printf 'a%.0s' $(seq 1000); printf END", yield_time_ms=10_000)
            path = content.split('path="')[1].split('"')[0]
            saved = await provider.run_command(f"cat {path}")
        assert "bytes omitted" in content
        assert content.count("a") < 200
        assert saved.stdout == "a" * 1000 + "END"

    async def test_command_longer_than_one_argument(self, provider: BackgroundCodeExecToolProvider) -> None:
        payload = "x" * 200_000
        async with provider as tools:
            code_exec, _ = _tools(tools)
            content = await _start(code_exec, f"s={payload}; echo ${{#s}}", yield_time_ms=10_000)
        assert "<exit_code>0</exit_code>" in content
        assert "200000" in content

    async def test_workdir(self, provider: BackgroundCodeExecToolProvider) -> None:
        async with provider as tools:
            code_exec, _ = _tools(tools)
            content = await _start(code_exec, "pwd", yield_time_ms=10_000, workdir="/tmp")
        assert "/tmp" in content

    async def test_min_wait_holds_a_poll_on_a_running_process(self) -> None:
        provider = BackgroundCodeExecToolProvider(HostBashProvider(), min_wait_sec=1)
        async with provider as tools:
            code_exec, _ = _tools(tools)
            content = await _start(code_exec, "sleep 30", yield_time_ms=0)
        assert "<yield_raised_to>1000</yield_raised_to>" in content

    async def test_unknown_process(self, provider: BackgroundCodeExecToolProvider) -> None:
        async with provider as tools:
            _, process_control = _tools(tools)
            result = await _run(process_control, ProcessControlParams(action="poll", process_id="proc_missing"))
        assert not result.success

    async def test_file_operations_delegate_to_backend(self, provider: BackgroundCodeExecToolProvider) -> None:
        async with provider:
            await provider.write_file_bytes("note.txt", b"hi")
            assert await provider.read_file_bytes("note.txt") == b"hi"
            assert await provider.file_exists("note.txt")

    def test_rejects_local_and_allowlisted_backends(self) -> None:
        with pytest.raises(TypeError):
            BackgroundCodeExecToolProvider(LocalCodeExecToolProvider())
        with pytest.raises(ValueError, match="allowed_commands"):
            BackgroundCodeExecToolProvider(HostBashProvider(allowed_commands=["echo"]))
