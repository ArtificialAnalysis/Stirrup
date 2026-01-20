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
            "Plan a software release with these tasks: Design (5 days), Backend (10 days, needs Design), "
            "Frontend (8 days, needs Design), Testing (4 days, needs Backend and Frontend), "
            "Documentation (3 days, can start after Backend). Two developers are available. "
            "What's the minimum time to complete? Output an Excel Gantt chart with the schedule."
        )


if __name__ == "__main__":
    asyncio.run(main())
# --8<-- [end:example]
