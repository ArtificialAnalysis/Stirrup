"""Shared tool middleware abstractions."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from functools import partial
from typing import cast

import anyio
from pydantic import BaseModel

from stirrup.core.models import Tool, ToolResult

__all__ = ["Call", "Sink", "ToolMiddleware", "call_executor"]

type Call = Callable[[BaseModel], Awaitable[ToolResult]]
"""Invokes the next layer of a wrapped tool with the given params."""


class _ProtectedText(str):
    pass


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
