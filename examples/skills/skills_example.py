"""Example: Using skills in an agent.

This example demonstrates how to use skills in an agent.
"""

import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider


async def main() -> None:
    """Run an agent that searches the web and creates a chart."""

    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
    )

    agent = Agent(
        client=client,
        name="agent",
        max_turns=20,
        tools=[DockerCodeExecToolProvider.from_dockerfile(dockerfile="examples/skills/Dockerfile")],
    )

    # Run with session context - handles tool lifecycle, logging and file outputs
    async with agent.session(
        input_files=["examples/skills/sample_data.csv"], output_dir="output/skills_example", skills_dir="skills"
    ) as session:
        _finish_params, _history, _metadata = await session.run(
            """
        Read the input sample_data.csv file and run full data analysis.
        """
        )


if __name__ == "__main__":
    asyncio.run(main())
