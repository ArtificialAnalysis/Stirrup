"""Middleware for spilling large tool output."""

from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel

from stirrup.core.models import Content, Tool, ToolResult
from stirrup.tools.middleware.middleware import Call, Sink, ToolMiddleware, _ProtectedText

if TYPE_CHECKING:
    from stirrup.tools.code_backends.base import CodeExecToolProvider

__all__ = ["DiskSpillMiddleware", "ExecEnvSink"]


class ExecEnvSink(Sink):
    """Writes dumps to an agent's execution environment."""

    def __init__(self, exec_env: "CodeExecToolProvider | None" = None, *, directory: str = "tool_outputs") -> None:
        """`exec_env` defaults to the one on the Agent's active session."""
        self._exec_env = exec_env
        self._directory = directory

    async def write(self, name: str, text: str) -> str:
        exec_env = self._exec_env or _session_exec_env()
        if exec_env is None:
            raise RuntimeError(
                "ExecEnvSink requires a CodeExecToolProvider. Either pass exec_env explicitly "
                "or include a CodeExecToolProvider in the Agent's tools list."
            )
        path = f"{self._directory}/{name}"
        await exec_env.write_file_bytes(path, text.encode())
        return path


def _session_exec_env() -> "CodeExecToolProvider | None":
    from stirrup.core.agent import _SESSION_STATE  # avoids a circular import

    state = _SESSION_STATE.get(None)
    return state.exec_env if state else None


class DiskSpillMiddleware(ToolMiddleware):
    """Spills oversized tool output to a sink."""

    def __init__(self, max_chars: int, sink: Sink, *, run_sync_in_thread: bool = True) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        super().__init__(run_sync_in_thread=run_sync_in_thread)
        self._max_chars = max_chars
        self._sink = sink

    async def handle(self, tool: Tool, params: BaseModel, call: Call) -> ToolResult:
        result = await call(params)
        text = "\n".join(_text_parts(result.content))
        if len(text) <= self._max_chars:
            return result

        path = await self._sink.write(_dump_name(tool.name), text)
        notice = _ProtectedText(f"Spilled {len(text)} characters to {path}.")
        content = [notice, result.content] if isinstance(result.content, str) else [notice, *result.content]
        return result.model_copy(update={"content": content})


def _text_parts(content: Content) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [block for block in content if isinstance(block, str)]


def _dump_name(tool_name: str) -> str:
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in tool_name)
    return f"{safe_name[:80] or 'tool'}_{uuid4().hex[:8]}.txt"
