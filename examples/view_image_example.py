"""Example: Web search and image viewing.

This example demonstrates an agent that can:
1. Search the web for images
2. Download images to the execution environment
3. View the downloaded images using ViewImageToolProvider
"""

import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider, ViewImageToolProvider, WebToolProvider


# --8<-- [start:main]
async def main() -> None:
    """Run an agent that can search the web and view images."""
    # Create client for OpenRouter
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
    )

    # ViewImageToolProvider will automatically use the Agent's CodeExecToolProvider
    agent = Agent(
        client=client,
        name="image_agent",
        tools=[
            LocalCodeExecToolProvider(),
            WebToolProvider(),
            ViewImageToolProvider(),  # Auto-detects exec_env
        ],
        max_turns=20,
    )

    async with agent.session(output_dir="output/view_image_example/") as session:
        _finish_params, _message_history, _run_metadata = await session.run(
            """Download an image of a kangaroo and describe what you see in it."""
        )


# --8<-- [end:main]

if __name__ == "__main__":
    asyncio.run(main())
