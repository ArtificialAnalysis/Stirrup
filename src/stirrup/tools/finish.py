from typing import Annotated

from pydantic import BaseModel, Field

from stirrup.core.models import FinishTool, FinishToolResult, ToolUseCountMetadata


class FinishParams(BaseModel):
    """Explanation for why the task is complete or cannot proceed."""

    reason: Annotated[str, Field(description="Reason for finishing.")]
    paths: Annotated[
        list[str], Field(description="List of file paths created or modified. Do not include directories, only files.")
    ]


SIMPLE_FINISH_TOOL: FinishTool[FinishParams, ToolUseCountMetadata] = FinishTool[FinishParams, ToolUseCountMetadata](
    description="Signal task completion with a reason. Use when the task is finished or cannot proceed further. Note that you will need a separate turn to finish.",
    parameters=FinishParams,
    executor=lambda params: FinishToolResult(
        content=params.reason, is_valid_finish_call=True, metadata=ToolUseCountMetadata()
    ),
)
