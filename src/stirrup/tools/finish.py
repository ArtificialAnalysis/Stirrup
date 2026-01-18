"""Simple finish tool with file existence validation."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from stirrup.constants import FINISH_TOOL_NAME
from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata


class FinishParams(BaseModel):
    """Explanation for why the task is complete or cannot proceed."""

    reason: Annotated[str, Field(description="Reason for finishing.")]
    paths: Annotated[
        list[str], Field(description="List of file paths created or modified. Do not include directories, only files.")
    ]


async def _validating_finish_executor(params: FinishParams) -> ToolResult[ToolUseCountMetadata]:
    """Validates all reported files exist before completing."""
    from stirrup.core.agent import _SESSION_STATE

    try:
        state = _SESSION_STATE.get(None)
        exec_env = state.exec_env if state else None
    except LookupError:
        exec_env = None

    if exec_env and params.paths:
        missing = [p for p in params.paths if not await exec_env.file_exists(p)]
        if missing:
            return ToolResult(
                content=f"ERROR: Files do not exist: {missing}. Verify paths and ensure files were saved.",
                metadata=ToolUseCountMetadata(),
                success=False,
            )

    return ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True)


SIMPLE_FINISH_TOOL: Tool[FinishParams, ToolUseCountMetadata] = Tool[FinishParams, ToolUseCountMetadata](
    name=FINISH_TOOL_NAME,
    description="Signal task completion with a reason. Use when the task is finished or cannot proceed further. Note that you will need a separate turn to finish.",
    parameters=FinishParams,
    executor=_validating_finish_executor,
)
