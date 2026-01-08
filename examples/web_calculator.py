"""Example: Web-enabled calculator agent with simplified session API.

This example demonstrates how to create an agent that can:
1. Perform calculations
2. Search the web (requires BRAVE_API_KEY)
3. Fetch web page content
"""

# --8<-- [start:setup]
import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import CALCULATOR_TOOL, DEFAULT_TOOLS

# Create client for OpenRouter
client = ChatCompletionsClient(
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4.5",
)

# Create agent with default tools + calculator tool
agent = Agent(
    client=client,
    name="web_calculator_agent",
    tools=[*DEFAULT_TOOLS, CALCULATOR_TOOL],
)
# --8<-- [end:setup]


# --8<-- [start:main]
async def main() -> None:
    """Run a simple web-enabled calculator agent."""
    # Create client for OpenRouter
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
        max_tokens=50_000,
    )

    # Create agent with default tools (coding env, web_search, web_fetch) + calculator tool
    tools = [*DEFAULT_TOOLS, CALCULATOR_TOOL]
    agent = Agent(
        client=client,
        name="web_calculator_agent",
        tools=tools,
        max_turns=10,
    )

    # Run with session context - handles all tool lifecycle and logging
    async with agent.session(resume=True) as session:
        _finish_params, _history, _metadata = await session.run(
            """Find the current world population and calculate what 10% of it would be.
            Use the web_search tool to find the current world population, then use
            the calculator to compute 10% of that number.
            When you're done, call the finish tool with your findings."""
        )


# --8<-- [end:main]

if __name__ == "__main__":
    asyncio.run(main())
