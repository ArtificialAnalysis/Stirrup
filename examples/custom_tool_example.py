"""Example: Creating a custom tool.

This example demonstrates how to define a custom tool with typed parameters
using Pydantic and register it with an agent.
"""

import asyncio

from pydantic import BaseModel, Field

from stirrup import Agent, Tool, ToolResult, ToolUseCountMetadata
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import DEFAULT_TOOLS


# --8<-- [start:tool]
class GreetParams(BaseModel):
    """Parameters for the greet tool."""

    name: str = Field(description="Name of the person to greet")
    formal: bool = Field(default=False, description="Use formal greeting")


def greet(params: GreetParams) -> ToolResult[ToolUseCountMetadata]:
    greeting = f"Good day, {params.name}." if params.formal else f"Hey {params.name}!"

    return ToolResult(
        content=greeting,
        metadata=ToolUseCountMetadata(),
    )


GREET_TOOL = Tool(
    name="greet",
    description="Greet someone by name",
    parameters=GreetParams,
    executor=greet,
)

# Create client for OpenRouter
client = ChatCompletionsClient(
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4.5",
)

# Add custom tool to default tools
agent = Agent(
    client=client,
    name="greeting_agent",
    tools=[*DEFAULT_TOOLS, GREET_TOOL],
)
# --8<-- [end:tool]


async def main() -> None:
    """Run an agent with a custom greeting tool."""
    async with agent.session() as session:
        await session.run("Please greet Alice formally, then greet Bob casually.")


if __name__ == "__main__":
    asyncio.run(main())
