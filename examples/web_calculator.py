"""Example: Web-enabled calculator agent with simplified session API.

This example demonstrates how to create an agent that can:
1. Perform calculations
2. Search the web (requires BRAVE_API_KEY)
3. Fetch web page content
"""

# --8<-- [start:setup]
import argparse
import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import CALCULATOR_TOOL
from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider
from stirrup.tools.web import WebToolProvider

DEFAULT_OPENROUTER_SLUG = "anthropic/claude-sonnet-4.5"

# Create client for OpenRouter
client = ChatCompletionsClient(
    base_url="https://openrouter.ai/api/v1",
    model=DEFAULT_OPENROUTER_SLUG,
)

# Create agent with E2B execution + web tools + calculator
# (This is just for the docs snippet above — the actual runnable code is in main() below)
agent = Agent(
    client=client,
    name="web_calculator_agent",
    tools=[E2BCodeExecToolProvider(), WebToolProvider(), CALCULATOR_TOOL],
)
# --8<-- [end:setup]


# --8<-- [start:main]
async def main(openrouter_slug: str) -> None:
    """Run a simple web-enabled calculator agent."""
    # Create client for OpenRouter
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model=openrouter_slug,
        max_tokens=50_000,
    )

    # Create agent with E2B execution + web tools + calculator
    tools = [E2BCodeExecToolProvider(), WebToolProvider(), CALCULATOR_TOOL]
    agent = Agent(
        client=client,
        name="web_calculator_agent",
        tools=tools,
        max_turns=10,
    )

    # Run with session context - handles all tool lifecycle and logging
    async with agent.session(output_dir="output") as session:
        _finish_params, _history, _metadata = await session.run(
            """Find the current world population and calculate what 10% of it would be.
            Use the web_search tool to find the current world population, then use
            the calculator to compute 10% of that number.
            When you're done, call the finish tool with your findings."""
        )


# --8<-- [end:main]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run the web calculator example with an OpenRouter model slug.")
    parser.add_argument(
        "openrouter_slug",
        nargs="?",
        default=DEFAULT_OPENROUTER_SLUG,
        help=f"OpenRouter model slug to use (default: {DEFAULT_OPENROUTER_SLUG})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args.openrouter_slug))
