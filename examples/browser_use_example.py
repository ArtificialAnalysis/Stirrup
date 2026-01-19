"""Example: Browser automation with BrowserUseToolProvider.

This example demonstrates how to use the BrowserUseToolProvider to automate
browser interactions. The agent can navigate pages, click elements, fill forms,
and extract information.

Prerequisites:
    - Install Chromium: `uvx browser-use install`
    - For cloud browser (optional): Set BROWSER_USE_API_KEY environment variable

Local browser usage (default) does not require an API key.
"""

import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import DEFAULT_TOOLS
from stirrup.tools.browser_use import BrowserUseToolProvider

# --8<-- [start:example]
# Create client (any LLM that supports tool calling)
client = ChatCompletionsClient(
    base_url="https://openrouter.ai/api/v1",
    model="anthropic/claude-sonnet-4.5",
)

# Create browser tool provider
# headless=False shows the browser window (useful for debugging)
browser_provider = BrowserUseToolProvider(
    headless=False,  # Set to True for headless mode
    # For cloud browser (optional):
    # use_cloud=True,  # Requires BROWSER_USE_API_KEY
)

# Create agent with browser tools
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
# --8<-- [end:example]


async def main() -> None:
    """Run browser automation example."""
    async with agent.session() as session:
        # Example: Search for something on Google
        await session.run(
            "Go to google.com, search for 'Stirrup AI agent framework', and tell me what the first result is about."
        )


if __name__ == "__main__":
    asyncio.run(main())
