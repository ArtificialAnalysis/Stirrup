import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import DEFAULT_TOOLS, USER_INPUT_TOOL


async def main() -> None:
    """Run an agent that searches the web and creates a chart."""

    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
    )

    agent = Agent(client=client, name="agent", tools=[*DEFAULT_TOOLS, USER_INPUT_TOOL], max_turns=15)

    # Run with session context - handles tool lifecycle, logging and file outputs
    async with agent.session(output_dir="output/getting_started_example") as session:
        _finish_params, _history, _metadata = await session.run(
            """
        What is the population of my home country over the last 3 years? Search the web to
        find out and create a chart using matplotlib showing the population per year.
        """
        )


if __name__ == "__main__":
    asyncio.run(main())
