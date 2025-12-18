"""Example: Using skills in an agent.

This example demonstrates how to use skills in an agent.
"""

import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider


# --8<-- [start:example]
async def main() -> None:
    """Run an agent with skills for data analysis."""
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

    async with agent.session(
        input_files=["examples/skills/sample_data.csv"],
        output_dir="output/skills_example",
        skills_dir="skills",
    ) as session:
        await session.run("Read the input sample_data.csv file and run full data analysis.")


# --8<-- [end:example]


if __name__ == "__main__":
    asyncio.run(main())
