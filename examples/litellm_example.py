"""Example: Using Agent with LiteLLM for multi-provider support.

Demonstrates using LiteLLM to connect to non-OpenAI providers like Anthropic Claude,
Google Gemini, etc. For LiteLLM, create the client directly and pass it to the Agent.

Requires:
- uv pip install stirrup[litellm]
- ANTHROPIC_API_KEY environment variable (for this example)
"""

import asyncio

from stirrup import Agent
from stirrup.clients.litellm_client import LiteLLMClient


async def main() -> None:
    """Run an agent using Anthropic Claude via LiteLLM."""
    # --8<-- [start:client]
    # Create LiteLLM client for Anthropic Claude
    # See https://docs.litellm.ai/docs/providers for all supported providers
    client = LiteLLMClient(model_slug="anthropic/claude-sonnet-4-5", max_tokens=64_000)

    # Pass client to Agent - model info comes from client.model_slug
    agent = Agent(
        client=client,
        name="claude_agent",
    )
    # --8<-- [end:client]

    async with agent.session(output_dir="./output/litellm_example") as session:
        await session.run(
            "What has the temperature been in the last 3 days in San Francisco? "
            "Provide a brief summary and output a pdf file with the summary and a graph."
        )


if __name__ == "__main__":
    asyncio.run(main())
