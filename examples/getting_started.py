"""Getting started example for Stirrup.

Demonstrates the core pattern:
1. Create a ChatCompletionsClient for your LLM provider
2. Create an Agent with the client
3. Run it in a session context
4. Agent searches the web and creates a chart as output
"""

# --8<-- [start:simple]
import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient


async def main() -> None:
    """Run an agent that searches the web and creates a chart."""

    # Create client using ChatCompletionsClient
    # Automatically uses OPENROUTER_API_KEY environment variable
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
    )

    # As no tools are provided, the agent will use the default tools, which consist of:
    # - Web tools (web search and web fetching, note web search requires BRAVE_API_KEY)
    # - Local code execution tool (to execute shell commands)
    agent = Agent(client=client, name="agent", max_turns=15)

    # Run with session context - handles tool lifecycle, logging and file outputs
    async with agent.session(output_dir="output/getting_started_example") as session:
        _finish_params, _history, _metadata = await session.run(
            """
        What is the population of the US over the last 3 years? Search the web to
        find out and create a chart using matplotlib showing the population per year.
        """
        )


if __name__ == "__main__":
    asyncio.run(main())
# --8<-- [end:simple]
