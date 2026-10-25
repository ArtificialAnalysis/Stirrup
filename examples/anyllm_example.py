"""Example: Using Agent with any-llm for multi-provider support.

Demonstrates using any-llm to connect to non-OpenAI providers like Anthropic Claude,
Google Gemini, etc. For any-llm, create the client directly and pass it to the Agent.

Requires:
- uv pip install stirrup[anyllm]
- ANTHROPIC_API_KEY environment variable (for this example)
"""

import asyncio

from stirrup import Agent
from stirrup.clients.anyllm_client import AnyLLMClient


async def main() -> None:
    """Run an agent using Anthropic Claude via any-llm."""
    # --8<-- [start:client]
    # Create any-llm client for Anthropic Claude
    # See https://mozilla-ai.github.io/any-llm/providers/ for all supported providers
    client = AnyLLMClient(
        model_slug="anthropic/claude-opus-5",
        max_tokens=8_192,
        context_window_tokens=1_000_000,
    )

    # Pass client to Agent - model info comes from client.model_slug
    agent = Agent(
        client=client,
        name="claude_agent",
    )
    # --8<-- [end:client]

    async with agent.session(output_dir="./output/anyllm_example") as session:
        await session.run(
            "What has the temperature been in the last 3 days in San Francisco? "
            "Provide a brief summary and output a pdf file with the summary and a graph."
        )


if __name__ == "__main__":
    asyncio.run(main())
