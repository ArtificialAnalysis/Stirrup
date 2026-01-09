"""User input tool for interactive clarification during agent execution.

This module provides the user_input tool that allows agents to ask questions
and receive text responses from users during task execution.

Example usage:
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.tools import DEFAULT_TOOLS, USER_INPUT_TOOL

    client = ChatCompletionsClient(model="gpt-5")
    agent = Agent(
        client=client,
        name="assistant",
        tools=[*DEFAULT_TOOLS, USER_INPUT_TOOL],
    )

    async with agent.session() as session:
        await session.run("Help me configure my project")
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from stirrup.core.models import Tool, ToolResult, ToolUseCountMetadata
from stirrup.utils.logging import AgentLoggerBase, console

__all__ = ["USER_INPUT_TOOL", "UserInputParams"]


class UserInputParams(BaseModel):
    """Parameters for asking the user a single question.

    Supports three question types:
    - "text": Free-form text input (default)
    - "choice": Multiple choice from a list of options
    - "confirm": Yes/no confirmation
    """

    question: Annotated[str, Field(description="A single question to ask the user (*not* multiple questions)")]
    question_type: Annotated[
        Literal["text", "choice", "confirm"],
        Field(
            default="text",
            description="Type of question: 'text' for free-form, 'choice' for multiple choice, 'confirm' for yes/no",
        ),
    ]
    choices: Annotated[
        list[str] | None,
        Field(default=None, description="List of valid choices (required when question_type is 'choice')"),
    ]
    default: Annotated[
        str,
        Field(default="", description="Default value if user presses Enter without input"),
    ]


def _get_logger() -> "AgentLoggerBase | None":
    """Get the current session's logger for pause/resume.

    Returns the logger from SessionState if available, None otherwise.
    """
    from stirrup.core.agent import _SESSION_STATE

    state = _SESSION_STATE.get(None)
    return state.logger if state else None


def user_input_executor(params: UserInputParams) -> ToolResult[ToolUseCountMetadata]:
    """Prompt the user for input and return their response."""
    logger = _get_logger()

    # Pause spinner before prompting
    if logger:
        logger.pause_live()

    try:
        # Print newline to separate from spinner, then display question in a styled panel
        console.print()
        panel = Panel(
            params.question,
            title="[bold cyan]🤔 Agent Question[/]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
        console.print(panel)

        # Get user input based on question type
        if params.question_type == "confirm":
            # Yes/no confirmation
            default_bool = params.default.lower() in ("yes", "y", "true", "1") if params.default else False
            result = Confirm.ask("[bold]Your answer[/]", default=default_bool, console=console)
            answer = "yes" if result else "no"

        elif params.question_type == "choice" and params.choices:
            # Multiple choice
            answer = Prompt.ask(
                "[bold]Your answer[/]",
                choices=params.choices,
                default=params.default,
                console=console,
            )

        else:
            # Free-form text (default)
            answer = Prompt.ask("[bold]Your answer[/]", default=params.default or "", console=console)

        return ToolResult(content=answer, metadata=ToolUseCountMetadata())

    finally:
        # Always resume spinner, even if an exception occurs
        if logger:
            logger.resume_live()


USER_INPUT_TOOL: Tool[UserInputParams, ToolUseCountMetadata] = Tool(
    name="user_input",
    description=(
        "Ask the user a question when you need clarification or are uncertain. "
        "Supports three types: 'text' for free-form input, 'choice' for multiple choice "
        "(provide choices list), 'confirm' for yes/no questions. Returns the user's response."
        "There should only EVER be one question per call to this tool."
        "If you need to ask multiple questions, you should call this tool multiple times."
    ),
    parameters=UserInputParams,
    executor=user_input_executor,
)
