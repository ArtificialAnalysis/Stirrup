"""Middleware for truncating large tool output."""

from pydantic import BaseModel

from stirrup.core.models import Content, Tool, ToolResult
from stirrup.tools.middleware.middleware import Call, ToolMiddleware, _ProtectedText

__all__ = ["ToolTruncatorMiddleware"]

_TRUNCATION_MARKER = "\n...[truncated]...\n"


class ToolTruncatorMiddleware(ToolMiddleware):
    """Truncates oversized tool output."""

    def __init__(
        self,
        max_chars: int,
        *,
        run_sync_in_thread: bool = True,
    ) -> None:
        if max_chars < len(_TRUNCATION_MARKER):
            raise ValueError(f"max_chars must be at least {len(_TRUNCATION_MARKER)}")
        super().__init__(run_sync_in_thread=run_sync_in_thread)
        self._max_chars = max_chars

    async def handle(self, tool: Tool, params: BaseModel, call: Call) -> ToolResult:  # noqa: ARG002
        result = await call(params)
        content = _truncate_content(result.content, self._max_chars)
        return result if content == result.content else result.model_copy(update={"content": content})


def _truncate_text(text: str, max_chars: int) -> str:
    remaining = max_chars - len(_TRUNCATION_MARKER)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining // 2
    return f"{text[:head_chars]}{_TRUNCATION_MARKER}{text[-tail_chars:] if tail_chars else ''}"


def _truncate_content(content: Content, max_chars: int) -> Content:
    if isinstance(content, str):
        return content if len(content) <= max_chars else _truncate_text(content, max_chars)

    protected_chars = sum(len(block) for block in content if isinstance(block, _ProtectedText))
    text_chars = sum(
        len(block) for block in content if isinstance(block, str) and not isinstance(block, _ProtectedText)
    )
    if protected_chars + text_chars <= max_chars:
        return content

    available = max_chars - protected_chars
    if available < len(_TRUNCATION_MARKER):
        raise ValueError("max_chars is too small to preserve the spill notice")

    remaining = available - len(_TRUNCATION_MARKER)
    head_chars = (remaining + 1) // 2
    tail_chars = remaining // 2
    tail_start = text_chars - tail_chars
    position = 0
    marker_added = False
    output = []

    for block in content:
        if isinstance(block, _ProtectedText) or not isinstance(block, str):
            output.append(block)
            continue

        start = position
        end = start + len(block)
        position = end
        parts: list[str] = []

        if start < head_chars:
            parts.append(block[: head_chars - start])
        if not marker_added and end > head_chars:
            parts.append(_TRUNCATION_MARKER)
            marker_added = True
        if end > tail_start:
            parts.append(block[max(0, tail_start - start) :])
        if parts:
            output.append("".join(parts))

    return output
