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


SIMPLE_FINISH_TOOL: Tool[FinishParams, ToolUseCountMetadata] = Tool[FinishParams, ToolUseCountMetadata](
    name=FINISH_TOOL_NAME,
    description="Signal task completion with a reason. Use when the task is finished or cannot proceed further. Note that you will need a separate turn to finish.",
    parameters=FinishParams,
    executor=lambda params: ToolResult(content=params.reason, metadata=ToolUseCountMetadata(), success=True),
)
