"""Example: Sub-agent delegation.

This example demonstrates how to create a supervisor agent that delegates tasks
to specialized sub-agents. The supervisor coordinates:
- A research sub-agent with web search and local code execution
- A report writing sub-agent with Docker-based code execution
"""

import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider, WebToolProvider
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider


async def main() -> None:
    """Run a supervisor agent that delegates tasks to sub-agents."""
    # Create client for OpenRouter (shared across all agents)
    client = ChatCompletionsClient(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-sonnet-4.5",
        max_tokens=64_000,
    )

    ## ------- define research sub-agent ------- ##
    # --8<-- [start:subagent]
    research_agent = Agent(
        client=client,
        name="research_sub_agent",
        tools=[WebToolProvider(), LocalCodeExecToolProvider()],
        max_turns=10,
        system_prompt=(
            "You are a research agent. When asked to complete research, save it all to a markdown file "
            "(using a code executor tool) and pass the filepath to the finish tool and mention it in the "
            "finish_reason. Remember you will need a turn to write the markdown file and a separate turn to finish."
        ),
    )

    # Convert agent to a tool for use by supervisor
    research_subagent_tool = research_agent.to_tool(
        description="Agent that can search the web and return the results.",
    )
    # --8<-- [end:subagent]

    ## ------- define report writing sub-agent ------- ##
    report_writing_agent = Agent(
        client=client,
        name="report_writing_sub_agent",
        tools=[DockerCodeExecToolProvider.from_image("ghcr.io/astral-sh/uv:python3.13-bookworm-slim")],
        max_turns=10,
    )

    report_writing_subagent_tool = report_writing_agent.to_tool(
        description="Create final reports using coding tools. Use uv to install any dependencies needed.",
    )

    ## ------- define supervisor agent ------- ##
    supervisor_agent = Agent(
        client=client,
        name="supervisor_agent",
        tools=[research_subagent_tool, report_writing_subagent_tool, LocalCodeExecToolProvider()],
        max_turns=5,
    )

    async with supervisor_agent.session(output_dir="output/sub_agent_example/") as session:
        _finish_params, _message_history, _run_metadata = await session.run(
            init_msgs="""
            Create a report on the latest hallucination benchmarks.
            Use your research sub-agent and report writing sub-agent to create the report.
            Output the report as a PDF file and pass the path in the finish tool.
            """
        )


if __name__ == "__main__":
    asyncio.run(main())
