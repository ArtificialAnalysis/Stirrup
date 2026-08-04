"""Use middleware to spill and truncate large tool output."""

import asyncio
from pathlib import Path

from pydantic import BaseModel, Field

from stirrup import (
    Agent,
    DiskSpillMiddleware,
    LocalDirSink,
    Tool,
    ToolResult,
    ToolTruncatorMiddleware,
    ToolUseCountMetadata,
)
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import CALCULATOR_TOOL


class DumpParams(BaseModel):
    """Parameters for the dump tool."""

    size: int = Field(default=5000, description="How many characters of filler to emit")


def dump_text(params: DumpParams) -> ToolResult[ToolUseCountMetadata]:
    return ToolResult(content="x" * params.size, metadata=ToolUseCountMetadata())


DUMP_TOOL = Tool(
    name="dump_text",
    description="Emit a long string of filler text (for demoing middleware)",
    parameters=DumpParams,
    executor=dump_text,
)

# --8<-- [start:middleware]
spill = DiskSpillMiddleware(
    max_chars=200,
    sink=LocalDirSink(Path("output/tool_outputs")),
)
truncate = ToolTruncatorMiddleware(max_chars=200)

# Spill before truncating so the file contains the full output
DUMP_TOOL_WITH_MIDDLEWARE = truncate(spill(DUMP_TOOL))
# --8<-- [end:middleware]

client = ChatCompletionsClient(
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4.5",
)

agent = Agent(
    client=client,
    name="middleware_agent",
    tools=[DUMP_TOOL_WITH_MIDDLEWARE, CALCULATOR_TOOL],
)


async def main() -> None:
    """Run an agent with middleware on one tool only."""
    async with agent.session(output_dir="output") as session:
        await session.run(
            "Call dump_text with size=5000, then use the calculator to compute 2+2. "
            "Finish with a short note about what dump_text returned."
        )


if __name__ == "__main__":
    asyncio.run(main())
