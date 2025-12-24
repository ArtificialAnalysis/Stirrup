from typing import Annotated

from pydantic import BaseModel, Field

from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata


class CalculatorParams(BaseModel):
    """Mathematical expression to be evaluated."""

    expression: Annotated[
        str,
        Field(description="Mathematical expression to evaluate (Python syntax, e.g., '2 + 2 * 3')"),
    ]


def calculator_executor(params: CalculatorParams) -> ToolResult[ToolUseCountMetadata]:
    """Evaluate mathematical expression in a limited eval environment."""
    try:
        # Safely evaluate the expression using Python's eval with restricted globals
        result = eval(params.expression, {"__builtins__": {}}, {})
        return ToolResult(content=f"Result: {result}", metadata=ToolUseCountMetadata())
    except Exception as e:
        return ToolResult(content=f"Error evaluating expression: {e!s}", success=False, metadata=ToolUseCountMetadata())


CALCULATOR_TOOL: Tool[CalculatorParams, ToolUseCountMetadata] = Tool[CalculatorParams, ToolUseCountMetadata](
    name="calculator",
    description="Evaluate mathematical expressions. Supports basic arithmetic operations (+, -, *, /, **, %, //).",
    parameters=CalculatorParams,
    executor=calculator_executor,  # ty: ignore[invalid-argument-type]
)
