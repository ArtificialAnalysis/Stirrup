"""Example: Code execution agent (E2B, Docker, or local).

This example demonstrates how to create an agent that can execute shell or Python
code in an isolated execution environment using one of several backends (E2B, Docker, or local).
You can switch between backends by commenting/uncommenting the relevant execution
environment instantiation.

Prerequisites for E2B:
- Set E2B_API_KEY environment variable.

See accompanying comments in the file for backend options.
"""

import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider


# --8<-- [start:main]
async def main() -> None:
    """Run a simple code execution agent."""
    # Create client for OpenRouter
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
    )

    # Choose your backend:
    code_exec_tool_provider = LocalCodeExecToolProvider()  # Local
    # code_exec_tool_provider = DockerCodeExecToolProvider.from_image("python:3.12-slim")  # Docker
    # code_exec_tool_provider = E2BCodeExecToolProvider()  # E2B (requires E2B_API_KEY)

    agent = Agent(
        client=client,
        name="code_executor_agent",
        tools=[code_exec_tool_provider],
        max_turns=20,
    )

    async with agent.session(
        input_files="examples/code_executor/task.txt",
        output_dir="output/code_executor_example/",
    ) as session:
        task = """
        You are a helpful coding assistant with access to a Python execution environment.
        Read the task from the input files and execute it. Use the code_exec tool to run the Python code.
        When you're done, call the finish tool with the results.
        """
        await session.run(task)


# --8<-- [end:main]

if __name__ == "__main__":
    asyncio.run(main())
