"""Example using the OpenResponses API client.

Demonstrates using the OpenResponsesClient, which uses OpenAI's Responses API
(POST /v1/responses) instead of the Chat Completions API. This API is useful
for providers that implement the newer Responses API format.

The Responses API has some differences from Chat Completions:
- System messages are passed as `instructions` parameter (separate from input)
- Uses `input` instead of `messages`
- Tool calls use `call_id` instead of `tool_call_id`
- Supports reasoning models with `reasoning_effort` parameter
"""

# --8<-- [start:example]
import asyncio

from stirrup import Agent
from stirrup.clients import OpenResponsesClient


async def main() -> None:
    """Run an agent using the OpenResponses API with a reasoning model."""

    # Create client using OpenResponsesClient
    # Uses the OpenAI Responses API (responses.create)
    # For reasoning models, you can set reasoning_effort
    client = OpenResponsesClient(
        model="gpt-5.2",
        reasoning_effort="medium",
    )

    agent = Agent(client=client, name="reasoning-agent", max_turns=19)

    async with agent.session(output_dir="output/open_responses_example") as session:
        _finish_params, _history, _metadata = await session.run(
            "Solve this step by step: If a train travels 120 miles in 2 hours, "
            "then stops for 30 minutes, then travels another 90 miles in 1.5 hours, "
            "what is its average speed for the entire journey including the stop?"
            "Output an excel document with the answer and the steps with formulas."
        )


if __name__ == "__main__":
    asyncio.run(main())
# --8<-- [end:example]
