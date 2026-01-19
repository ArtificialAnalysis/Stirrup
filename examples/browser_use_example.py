"""Example: Browser automation with BrowserUseToolProvider.

This example demonstrates how to use the BrowserUseToolProvider to automate
browser interactions. The agent can navigate pages, click elements, fill forms,
and extract information.

Prerequisites:
    - Install browser extra: `pip install 'stirrup[browser]'` or `uv add 'stirrup[browser]'`
    - Install Chromium: `uvx browser-use install`
    - For cloud browser (optional): Set BROWSER_USE_API_KEY environment variable
"""

# --8<-- [start:main]
import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import DEFAULT_TOOLS
from stirrup.tools.browser_use import BrowserUseToolProvider


async def main() -> None:
    """Run browser automation example."""
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
    )

    browser_provider = BrowserUseToolProvider(
        headless=False,  # Set to True for headless mode
    )

    agent = Agent(
        client=client,
        name="browser_agent",
        tools=[*DEFAULT_TOOLS, browser_provider],
        system_prompt=(
            "You are a web automation assistant. Use the browser tools to complete tasks. "
            "Always start by taking a snapshot to see the current page state and element indices. "
            "Use the indices from the snapshot when clicking or typing."
        ),
    )

    async with agent.session(output_dir="output/browser_use_example") as session:
        _finish_params, _history, _metadata = await session.run(
            "Go to artificial analysis and select o3 on the AA Index score"
        )


if __name__ == "__main__":
    asyncio.run(main())
# --8<-- [end:main]
