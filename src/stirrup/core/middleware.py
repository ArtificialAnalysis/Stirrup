"""Composable middleware for tools."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import anyio
from pydantic import BaseModel

from stirrup.core.models import Content, Tool, ToolResult

if TYPE_CHECKING:
    from stirrup.tools.code_backends.base import CodeExecToolProvider

__all__ = [
    "Call",
    "DiskSpillMiddleware",
    "ExecEnvSink",
    "LocalDirSink",
    "Sink",
    "Summarizer",
    "ToolMiddleware",
    "ToolTruncatorMiddleware",
    "call_executor",
]

type Call = Callable[[BaseModel], Awaitable[ToolResult]]
"""Invokes the next layer of a wrapped tool with the given params."""

type Summarizer = Callable[[str, int], str]
"""Shrinks text to fit within a character budget."""

_TRUNCATION_MARKER = "\n...[truncated]...\n"


def _truncate_text(text: str, max_chars: int) -> str:
    remaining = max_chars - len(_TRUNCATION_MARKER)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining // 2
    return f"{text[:head_chars]}{_TRUNCATION_MARKER}{text[-tail_chars:] if tail_chars else ''}"


async def call_executor(tool: Tool, params: BaseModel, *, run_sync_in_thread: bool = True) -> ToolResult:
    """Invoke a tool executor."""
    if inspect.iscoroutinefunction(tool.executor):
        return await tool.executor(params)
    if run_sync_in_thread:
        result = await anyio.to_thread.run_sync(tool.executor, params)  # ty: ignore[unresolved-attribute]
    else:
        result = tool.executor(params)
    return cast(ToolResult, await result) if inspect.isawaitable(result) else result


class ToolMiddleware(ABC):
    """Wrap a tool's executor without changing its schema."""

    _run_sync_in_thread = True

    def __init__(self, *, run_sync_in_thread: bool = True) -> None:
        self._run_sync_in_thread = run_sync_in_thread

    def __call__(self, tool: Tool) -> Tool:
        async def executor(params: BaseModel) -> ToolResult:
            call = partial(call_executor, tool, run_sync_in_thread=self._run_sync_in_thread)
            return await self.handle(tool, params, call)

        return tool.model_copy(update={"executor": executor})

    @abstractmethod
    async def handle(self, tool: Tool, params: BaseModel, call: Call) -> ToolResult:
        """Handle a call to the wrapped tool."""


class Sink(ABC):
    """Destination for dumped tool output."""

    @abstractmethod
    async def write(self, name: str, text: str) -> str:
        """Store `text` and return a path the agent can refer to."""


class LocalDirSink(Sink):
    """Writes dumps to a local directory."""

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)

    async def write(self, name: str, text: str) -> str:
        await anyio.Path(self._directory).mkdir(parents=True, exist_ok=True)
        path = self._directory / name
        await anyio.Path(path).write_text(text, encoding="utf-8")
        return str(path)


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
        notice = f"Output was {len(text)} characters and has been saved to {path}. Read that file to see it in full."
        content = [notice, result.content] if isinstance(result.content, str) else [notice, *result.content]
        return result.model_copy(update={"content": content})


class ToolTruncatorMiddleware(ToolMiddleware):
    """Truncates oversized tool output."""

    def __init__(
        self,
        max_chars: int,
        *,
        summarize: Summarizer = _truncate_text,
        run_sync_in_thread: bool = True,
    ) -> None:
        if max_chars < len(_TRUNCATION_MARKER):
            raise ValueError(f"max_chars must be at least {len(_TRUNCATION_MARKER)}")
        super().__init__(run_sync_in_thread=run_sync_in_thread)
        self._max_chars = max_chars
        self._summarize = summarize

    async def handle(self, tool: Tool, params: BaseModel, call: Call) -> ToolResult:  # noqa: ARG002
        result = await call(params)
        content = _truncate_content(result.content, self._max_chars, self._summarize)
        return result if content == result.content else result.model_copy(update={"content": content})


def _text_parts(content: Content) -> list[str]:
    if isinstance(content, str):
        return [content]
    return [block for block in content if isinstance(block, str)]


def _truncate_content(content: Content, max_chars: int, summarize: Summarizer) -> Content:
    if isinstance(content, str):
        return _summarize(content, max_chars, summarize)
    return [_summarize(block, max_chars, summarize) if isinstance(block, str) else block for block in content]


def _summarize(text: str, max_chars: int, summarize: Summarizer) -> str:
    if len(text) <= max_chars:
        return text
    result = summarize(text, max_chars)
    if len(result) > max_chars:
        raise ValueError("summarizer returned more than max_chars")
    return result


def _dump_name(tool_name: str) -> str:
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in tool_name)
    return f"{safe_name[:80] or 'tool'}_{uuid4().hex[:8]}.txt"
