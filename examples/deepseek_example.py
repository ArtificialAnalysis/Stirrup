"""Example: Using Agent with Deepseek's OpenAI-compatible API.

Demonstrates connecting to Deepseek using ChatCompletionsClient with a custom base_url.
This same pattern works for any OpenAI-compatible API (vLLM, Ollama, Azure OpenAI,
local models, etc.).

Requires: DEEPSEEK_API_KEY environment variable
"""

import asyncio
import os

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient


async def main() -> None:
    """Run an agent using Deepseek's API."""
    # --8<-- [start:client]
    # Create client using Deepseek's OpenAI-compatible endpoint
    client = ChatCompletionsClient(
        base_url="https://api.deepseek.com",
        model="deepseek-chat",  # or "deepseek-reasoner" for R1
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )

    agent = Agent(client=client, name="deepseek_agent")
    # --8<-- [end:client]

    async with agent.session(output_dir="./output") as session:
        await session.run("What is 2 + 2? Explain your reasoning step by step.")


if __name__ == "__main__":
    asyncio.run(main())
