"""CLI entry point for running Stirrup agents from the command line."""

import argparse
import asyncio
import sys
from typing import Any

from stirrup import Agent, Tool, ToolProvider
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import DEFAULT_TOOLS, USER_INPUT_TOOL
from stirrup.tools.view_image import ViewImageToolProvider


def _build_tools(args: argparse.Namespace) -> list[Tool[Any, Any] | ToolProvider]:
    """Build the tools list from CLI arguments."""
    tools: list[Tool[Any, Any] | ToolProvider] = [
        *DEFAULT_TOOLS,
        USER_INPUT_TOOL,
        ViewImageToolProvider(),
    ]

    if args.mcp_config:
        from stirrup.tools.mcp import MCPToolProvider

        tools.append(MCPToolProvider.from_config(args.mcp_config))

    return tools


def main() -> None:
    """Run a Stirrup agent from the command line."""
    parser = argparse.ArgumentParser(
        prog="stirrup",
        description="Run a Stirrup agent from the command line.",
    )
    parser.add_argument("--task", required=True, help="Task description for the agent")
    parser.add_argument(
        "--model", default="anthropic/claude-sonnet-4.5", help="Model identifier (default: anthropic/claude-sonnet-4.5)"
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="API base URL (default: https://openrouter.ai/api/v1)",
    )
    parser.add_argument("--api-key", default=None, help="API key (default: OPENROUTER_API_KEY env var)")
    parser.add_argument("--output-dir", default="./output", help="Directory for agent output files (default: ./output)")
    parser.add_argument("--max-turns", type=int, default=50, help="Max agent loop iterations (default: 50)")
    parser.add_argument("--system-prompt", default=None, help="Custom system prompt")
    parser.add_argument("--input-files", nargs="+", default=None, help="Input files (space-separated, supports globs)")
    parser.add_argument("--skills-dir", default=None, help="Path to skills directory")
    parser.add_argument("--mcp-config", default=None, help="Path to MCP config file (mcp.json)")

    args = parser.parse_args()

    client = ChatCompletionsClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )

    agent = Agent(
        client=client,
        name="cli-agent",
        max_turns=args.max_turns,
        system_prompt=args.system_prompt,
        tools=_build_tools(args),
    )

    async def _run() -> None:
        async with agent.session(
            output_dir=args.output_dir,
            input_files=args.input_files,
            skills_dir=args.skills_dir,
        ) as session:
            finish_params, _history, _metadata = await session.run(args.task)
            if finish_params:
                print(f"Finish reason: {finish_params.reason}")
            else:
                print("Agent did not finish (max turns reached).", file=sys.stderr)
                sys.exit(1)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
