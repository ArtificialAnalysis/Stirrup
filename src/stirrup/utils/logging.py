"""Rich logging for agent workflows with visual hierarchy."""

import html
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Self, cast

from pydantic import BaseModel
from rich import box
from rich.console import Console, RenderableType
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from stirrup.core.models import AssistantMessage, ToolMessage, UserMessage, _aggregate_list, aggregate_metadata

__all__ = [
    "AgentLogger",
    "AgentLoggerBase",
    "console",
]

# Shared console instance
console = Console()

# Indentation spaces per sub-agent nesting level
SUBAGENT_INDENT_SPACES: int = 8


def _is_subagent_metadata(data: object) -> bool:
    """Check if data represents sub-agent metadata.

    Sub-agent metadata can be:
    - A Pydantic SubAgentMetadata object with run_metadata attribute
    - A dict where all values are dicts/objects (from aggregate_metadata flattening)
    """
    # Check for Pydantic SubAgentMetadata object
    if hasattr(data, "run_metadata") and isinstance(data.run_metadata, dict):
        return True
    # Check for flattened dict of dicts (from aggregate_metadata)
    if isinstance(data, dict) and data:
        return all(isinstance(v, dict) or hasattr(v, "model_dump") for v in data.values())
    return False


def _format_token_usage(data: object) -> str:
    """Format token_usage (dict or TokenUsage object) as a human-readable string."""
    if isinstance(data, dict):
        # Dict representation
        data_dict = cast(dict[str, Any], data)
        input_tokens: int = data_dict.get("input", 0)
        output_tokens: int = data_dict.get("output", 0)
        reasoning_tokens: int = data_dict.get("reasoning", 0)
    elif hasattr(data, "input") and hasattr(data, "output"):
        # Pydantic TokenUsage object - use getattr for type safety
        input_tokens = int(getattr(data, "input", 0))
        output_tokens = int(getattr(data, "output", 0))
        reasoning_tokens = int(getattr(data, "reasoning", 0))
    else:
        return str(data)
    total = input_tokens + output_tokens + reasoning_tokens
    return f"{total:,} tokens"


def _get_nested_tools(data: object) -> dict[str, object]:
    """Extract nested tools dict from sub-agent metadata."""
    if hasattr(data, "run_metadata"):
        # Pydantic SubAgentMetadata - return its run_metadata
        run_metadata = data.run_metadata
        if isinstance(run_metadata, dict):
            return cast(dict[str, object], run_metadata)
    if isinstance(data, dict):
        # Already a dict
        return cast(dict[str, object], data)
    return {}


def _add_tool_branch(
    parent: Tree,
    tool_name: str,
    tool_data: object,
    skip_fields: set[str],
) -> None:
    """Add a tool entry to the tree, handling nested sub-agent data recursively.

    Args:
        parent: The tree or branch to add to
        tool_name: Name of the tool or sub-agent
        tool_data: The tool's metadata (dict, Pydantic model, list, or scalar)
        skip_fields: Fields to skip when displaying dict contents
    """
    # Special case: token_usage formatted as total tokens
    if tool_name == "token_usage":
        if isinstance(tool_data, list) and tool_data:
            parent.add(f"[dim]token_usage:[/] {_format_token_usage(tool_data[0])}")
        else:
            parent.add(f"[dim]token_usage:[/] {_format_token_usage(tool_data)}")
        return

    # Case 1: List → aggregate using __add__, then recurse
    if isinstance(tool_data, list) and tool_data:
        aggregated = _aggregate_list(tool_data)
        if aggregated is not None:
            _add_tool_branch(parent, tool_name, aggregated, skip_fields)
        return

    # Case 2: SubAgentMetadata → recurse into run_metadata only
    if _is_subagent_metadata(tool_data):
        branch = parent.add(f"[magenta]{tool_name}[/]")
        for nested_name, nested_data in sorted(_get_nested_tools(tool_data).items()):
            _add_tool_branch(branch, nested_name, nested_data, skip_fields)
        return

    # Case 3: Leaf node - display fields as branches
    # Convert to dict if Pydantic model
    if hasattr(tool_data, "model_dump"):
        data_dict = cast(Callable[[], dict[str, Any]], tool_data.model_dump)()
    elif isinstance(tool_data, dict):
        data_dict = cast(dict[str, Any], tool_data)
    else:
        # Scalar value - just show it inline
        parent.add(f"[magenta]{tool_name}[/]: {tool_data}")
        return

    # Show num_uses inline with the tool name if present
    num_uses = data_dict.get("num_uses")
    if num_uses is not None:
        branch = parent.add(f"[magenta]{tool_name}[/]: {num_uses} call(s)")
    else:
        branch = parent.add(f"[magenta]{tool_name}[/]")

    for k, v in data_dict.items():
        if k not in skip_fields and v is not None:
            branch.add(f"[dim]{k}:[/] {v}")


class AgentLoggerBase(ABC):
    """Abstract base class for agent loggers.

    Defines the interface that Agent uses for logging. Implement this to create
    custom loggers (e.g., for testing, file output, or monitoring services).

    Properties are set by Agent after construction:
    - name, model, max_turns, depth: Agent configuration
    - finish_params, run_metadata, output_dir: Set before __exit__ for final stats
    """

    # Properties set by Agent after construction
    name: str
    model: str | None
    max_turns: int | None
    depth: int

    # State updated during run (set before __exit__)
    finish_params: BaseModel | None
    run_metadata: dict[str, list[Any]] | None
    output_dir: str | None

    @abstractmethod
    def __enter__(self) -> Self:
        """Enter logging context. Called when agent session starts."""
        ...

    @abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit logging context. Called when agent session ends."""
        ...

    @abstractmethod
    def on_step(
        self,
        step: int,
        tool_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Report step progress and stats during agent execution."""
        ...

    @abstractmethod
    def assistant_message(
        self,
        turn: int,
        max_turns: int,
        assistant_message: AssistantMessage,
    ) -> None:
        """Log an assistant message."""
        ...

    @abstractmethod
    def user_message(self, user_message: UserMessage) -> None:
        """Log a user message."""
        ...

    @abstractmethod
    def task_message(self, task: str | list[Any]) -> None:
        """Log the initial task/prompt at the start of a run."""
        ...

    @abstractmethod
    def tool_result(self, tool_message: ToolMessage) -> None:
        """Log a tool execution result."""
        ...

    @abstractmethod
    def context_summarization_start(self, pct_used: float, cutoff: float) -> None:
        """Log that context summarization is starting."""
        ...

    @abstractmethod
    def context_summarization_complete(self, summary: str, bridge: str) -> None:
        """Log completed context summarization."""
        ...

    # Standard logging methods
    @abstractmethod
    def debug(self, message: str, *args: object) -> None:
        """Log a debug message."""
        ...

    @abstractmethod
    def info(self, message: str, *args: object) -> None:
        """Log an info message."""
        ...

    @abstractmethod
    def warning(self, message: str, *args: object) -> None:
        """Log a warning message."""
        ...

    @abstractmethod
    def error(self, message: str, *args: object) -> None:
        """Log an error message."""
        ...

    def pause_live(self) -> None:  # noqa: B027
        """Pause live display (e.g., spinner) before user interaction."""

    def resume_live(self) -> None:  # noqa: B027
        """Resume live display after user interaction."""


class AgentLogger(AgentLoggerBase):
    """Rich console logger for agent workflows.

    Implements AgentLoggerBase with rich formatting, spinners, and visual hierarchy.
    Each agent (including sub-agents) should have its own logger instance.

    Usage:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        # Agent creates logger internally by default
        client = ChatCompletionsClient(model="gpt-4")
        agent = Agent(client=client, name="assistant")

        # Or pass a pre-configured logger
        logger = AgentLogger(show_spinner=False)
        agent = Agent(client=client, name="assistant", logger=logger)

        # Agent sets these properties before calling __enter__:
        # logger.name, logger.model, logger.max_turns, logger.depth

        # Agent sets these before calling __exit__:
        # logger.finish_params, logger.run_metadata, logger.output_dir
    """

    def __init__(
        self,
        *,
        show_spinner: bool = True,
        level: int = logging.INFO,
    ) -> None:
        """Initialize the agent logger.

        Args:
            show_spinner: Whether to show a spinner while agent runs (only for depth=0)
            level: Logging level (default: INFO)
        """
        # Properties set by Agent before __enter__
        self.name: str = "agent"
        self.model: str | None = None
        self.max_turns: int | None = None
        self.depth: int = 0

        # State set by Agent before __exit__
        self.finish_params: BaseModel | None = None
        self.run_metadata: dict[str, list[Any]] | None = None
        self.output_dir: str | None = None

        # Configuration
        self._show_spinner = show_spinner
        self._level = level

        # Spinner state (only used when depth == 0 and show_spinner is True)
        self._current_step = 0
        self._tool_calls = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._live: Live | None = None

        # Configure rich logging on first logger creation
        self._configure_logging()

    def _configure_logging(self) -> None:
        """Configure rich logging with agent-aware formatting."""
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
            show_level=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))

        root = logging.getLogger()
        root.handlers.clear()
        root.addHandler(handler)
        root.setLevel(self._level)

        # Silence noisy loggers
        for name in [
            "LiteLLM",
            "httpx",
            "httpcore",
            "openai",
            "utils",
            "asyncio",
            "filelock",
            "fsspec",
            "urllib3",
            "markdown_it",
            "docker",
            "e2b",
            "e2b.api",
            "e2b.sandbox_async",
            "e2b.sandbox_sync",
        ]:
            logging.getLogger(name).setLevel(logging.WARNING)

        # These loggers need ERROR level to fully suppress verbose debug output
        for name in [
            "mcp",
            "mcp.client",
            "mcp.client.sse",
            "mcp.client.streamable_http",
            "httpx_sse",
            "readability",
            "readability.readability",
            "trafilatura",
            "trafilatura.core",
            "trafilatura.readability_lxml",
            "htmldate",
            "courlan",
        ]:
            logging.getLogger(name).setLevel(logging.ERROR)

    def _get_indent(self) -> str:
        """Get indentation string based on current agent depth."""
        return "│   " * self.depth

    def _print_indented(self, renderable: RenderableType, indent: str | int) -> None:
        """Print a renderable with indentation using Padding.

        Args:
            renderable: The Rich renderable to print (Panel, Tree, Table, Rule, etc.)
            indent: Either a string prefix or number of spaces for left padding
        """
        if isinstance(indent, str):
            # For string indents, use capture method
            with console.capture() as capture:
                console.print(renderable)
            output = capture.get()
            for line in output.rstrip("\n").split("\n"):
                console.print(f"{indent}{line}")
        else:
            # For numeric indents, use Padding
            console.print(Padding(renderable, (0, 0, 0, indent)))

    def _make_spinner_text(self) -> Text:
        """Create styled text for the spinner display."""
        text = Text()

        text.append("Running ", style="bold green")
        text.append(self.name, style="bold green")

        # Separator
        text.append("  │  ", style="dim")

        # Step count
        if self.max_turns:
            text.append(f"{self._current_step}/{self.max_turns}", style="cyan bold")
            text.append(" steps", style="cyan")
        else:
            text.append(f"{self._current_step}", style="cyan bold")
            text.append(" steps", style="cyan")

        # Separator
        text.append("  │  ", style="dim")

        # Tool calls
        text.append(f"{self._tool_calls}", style="magenta bold")
        text.append(" tool calls", style="magenta")

        # Separator
        text.append("  │  ", style="dim")

        # Input tokens
        text.append(f"{self._input_tokens:,}", style="yellow bold")
        text.append(" input tokens", style="yellow")

        # Separator
        text.append("  │  ", style="dim")

        # Output tokens
        text.append(f"{self._output_tokens:,}", style="blue bold")
        text.append(" output tokens", style="blue")

        return text

    def _make_spinner(self) -> Spinner:
        """Create spinner with current stats."""
        return Spinner("aesthetic", text=self._make_spinner_text(), style="green")

    # -------------------------------------------------------------------------
    # Context Manager Methods (AgentLoggerBase implementation)
    # -------------------------------------------------------------------------

    def __enter__(self) -> Self:
        """Enter logging context. Logs agent start and starts spinner if depth=0."""
        # Log agent start (rule + system prompt display)
        indent_spaces = self.depth * SUBAGENT_INDENT_SPACES

        # Build title with optional model info
        model_str = f" ({self.model})" if self.model else ""
        if self.depth == 0:
            title = f"▶ {self.name}{model_str}"
            console.rule(f"[bold cyan]{title}[/]", style="cyan")
        else:
            title = f"▶ {self.name}: Level {self.depth}{model_str}"
            rule = Rule(f"[bold cyan]{title}[/]", style="cyan")
            self._print_indented(rule, indent_spaces)
        console.print()

        # Start spinner only for top-level agent
        if self.depth == 0 and self._show_spinner:
            self._live = Live(self._make_spinner(), console=console, refresh_per_second=10)
            self._live.start()

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit logging context. Stops spinner and logs completion stats."""
        # Stop spinner first
        if self._live:
            self._live.stop()
            self._live = None

        error = str(exc_val) if exc_type is not None else None
        self._log_finish(error=error)

    def _log_finish(self, error: str | None = None) -> None:
        """Log agent completion with full statistics."""
        console.print()  # Add spacing before finish

        # Determine status
        if error:
            status = f"[bold red]✗ {self.name} - Error[/]"
            style = "red"
        elif self.finish_params is None:
            # Agent didn't call finish tool (e.g., ran out of turns)
            status = f"[bold red]✗ {self.name} - Failed[/]"
            style = "red"
        else:
            status = f"[bold green]✓ {self.name} - Complete[/]"
            style = "green"

        indent_spaces = self.depth * SUBAGENT_INDENT_SPACES
        if self.depth == 0:
            console.rule(status, style=style)
        else:
            rule = Rule(status, style=style)
            self._print_indented(rule, indent_spaces)

        # Display error if present
        if error:
            error_text = Text(f"Error: {error}", style="red")
            if self.depth > 0:
                self._print_indented(error_text, indent_spaces)
            else:
                console.print(error_text)
            console.print()

        # For subagents, only show the status rule (and error if present)
        if self.depth > 0:
            return

        # Extract paths from finish_params for use in metadata section
        paths = None
        if self.finish_params:
            params = self.finish_params.model_dump()
            reason = params.get("reason", "")
            paths = params.get("paths")

            # Reason panel as markdown (full width)
            reason_panel = Panel(
                Markdown(reason) if reason else "[dim]No reason provided[/]",
                title="[bold]Reason[/]",
                title_align="left",
                border_style="cyan",
                expand=True,
            )
            console.print(reason_panel)
            console.print()

        # Display run metadata statistics and paths in 1:1:1 layout
        if self.run_metadata or paths:
            # Aggregate metadata to roll up sub-agent token usage into the total
            if self.run_metadata:
                aggregated = aggregate_metadata(self.run_metadata, return_json_serializable=False)
                token_usage_list = aggregated.get("token_usage", [])
            else:
                token_usage_list = []
            tool_keys = (
                [k for k in self.run_metadata if k not in ("token_usage", "finish")] if self.run_metadata else []
            )

            # Build tool usage tree
            tool_panel = None
            if tool_keys and self.run_metadata:
                tool_tree = Tree("🔧 [bold]Tools[/]", guide_style="dim")
                skip_fields = {"num_uses"}
                for tool_name in sorted(tool_keys):
                    _add_tool_branch(tool_tree, tool_name, self.run_metadata[tool_name], skip_fields)

                tool_panel = Panel(
                    tool_tree,
                    title="[bold]Tool Usage[/]",
                    title_align="left",
                    border_style="magenta",
                    expand=True,
                )

            # Build paths panel
            paths_panel = None
            if paths:
                paths_tree = Tree("📁 [bold]Files[/]", guide_style="dim")
                # If output_dir is provided, add it as a parent node
                if self.output_dir:
                    output_branch = paths_tree.add(f"[magenta]{self.output_dir}/[/]")
                    for path in paths:
                        output_branch.add(f"[green]{path}[/]")
                else:
                    for path in paths:
                        paths_tree.add(f"[green]{path}[/]")

                paths_panel = Panel(
                    paths_tree,
                    title="[bold]Paths[/]",
                    title_align="left",
                    border_style="cyan",
                    expand=True,
                )

            # Build token usage table
            token_panel = None
            if token_usage_list:
                total_input = sum(getattr(u, "input", 0) for u in token_usage_list)
                total_output = sum(getattr(u, "output", 0) for u in token_usage_list)
                total_reasoning = sum(getattr(u, "reasoning", 0) for u in token_usage_list)
                total_tokens = sum(getattr(u, "total", 0) for u in token_usage_list)

                token_table = Table(
                    box=box.SIMPLE,
                    show_header=True,
                    header_style="bold",
                    show_footer=True,
                    expand=True,
                )
                token_table.add_column("Type", style="cyan", footer="[bold]Total[/]")
                token_table.add_column("Count", justify="right", style="green", footer=f"[bold]{total_tokens:,}[/]")

                token_table.add_row("Input", f"{total_input:,}")
                token_table.add_row("Output", f"{total_output:,}")
                if total_reasoning > 0:
                    token_table.add_row("Reasoning", f"{total_reasoning:,}")

                token_panel = Panel(
                    token_table,
                    title="[bold]Token Usage[/]",
                    title_align="left",
                    border_style="green",
                    expand=True,
                )

            # Display panels in 1:1:1 ratio layout (Tool Usage | Paths | Token Usage)
            panels = [p for p in [tool_panel, paths_panel, token_panel] if p is not None]
            if panels:
                layout_table = Table.grid(expand=True)
                for _ in panels:
                    layout_table.add_column(ratio=1)
                layout_table.add_row(*panels)
                console.print(layout_table)
                console.print()

        console.rule(style="dim")

        # Display max turns exceeded error panel (only for top-level agent, and only if no other error)
        if self.finish_params is None and self.max_turns is not None and error is None:
            content = Text()
            content.append("Maximum turns reached\n\n", style="bold")
            content.append("Turns used: ", style="dim")
            content.append(f"{self.max_turns}", style="bold red")
            content.append("\n\n")
            content.append(
                "The agent was not able to finish the task. Consider increasing the max_turns parameter.",
                style="italic",
            )

            panel = Panel(
                content,
                title="[bold red]⚠ Max Turns Exceeded[/]",
                title_align="left",
                border_style="red",
                padding=(0, 1),
            )
            console.print(panel)
            console.print()

    def on_step(
        self,
        step: int,
        tool_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Report step progress and stats during agent execution."""
        self._current_step = step
        self._tool_calls = tool_calls
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        if self._live:
            self._live.update(self._make_spinner())

    def pause_live(self) -> None:
        """Pause the live spinner display.

        Call this before prompting for user input to prevent the spinner
        from interfering with the input prompt.
        """
        if self._live is not None:
            self._live.stop()

    def resume_live(self) -> None:
        """Resume the live spinner display.

        Call this after user input is complete to restart the spinner.
        """
        if self._live is not None:
            self._live.start()

    def set_level(self, level: int) -> None:
        """Set the logging level."""
        self._level = level
        # Also update root logger level
        logging.getLogger().setLevel(level)

    def is_enabled_for(self, level: int) -> bool:
        """Check if a given log level is enabled."""
        return level >= self._level

    # -------------------------------------------------------------------------
    # Standard logging methods (debug, info, warning, error, critical)
    # -------------------------------------------------------------------------

    def debug(self, message: str, *args: object) -> None:
        """Log a debug message (dim style)."""
        if self._level <= logging.DEBUG:
            formatted = message % args if args else message
            console.print(f"[dim]{formatted}[/]")

    def info(self, message: str, *args: object) -> None:
        """Log an info message."""
        if self._level <= logging.INFO:
            formatted = message % args if args else message
            console.print(formatted)

    def warning(self, message: str, *args: object) -> None:
        """Log a warning message (yellow style)."""
        if self._level <= logging.WARNING:
            formatted = message % args if args else message
            console.print(f"[yellow]⚠ {formatted}[/]")

    def error(self, message: str, *args: object) -> None:
        """Log an error message (red style)."""
        if self._level <= logging.ERROR:
            formatted = message % args if args else message
            console.print(f"[red]✗ {formatted}[/]")

    def critical(self, message: str, *args: object) -> None:
        """Log a critical message (bold red style)."""
        if self._level <= logging.CRITICAL:
            formatted = message % args if args else message
            console.print(f"[bold red]✗ CRITICAL: {formatted}[/]")

    def exception(self, message: str, *args: object) -> None:
        """Log an error message with exception traceback (red style with traceback)."""
        if self._level <= logging.ERROR:
            formatted = message % args if args else message
            console.print(f"[red]✗ {formatted}[/]")
            console.print_exception()

    # -------------------------------------------------------------------------
    # Message Logging Methods (AgentLoggerBase implementation)
    # -------------------------------------------------------------------------

    def assistant_message(
        self,
        turn: int,
        max_turns: int,
        assistant_message: AssistantMessage,
    ) -> None:
        """Log an assistant message with content and tool calls in a panel.

        Args:
            turn: Current turn number (1-indexed)
            max_turns: Maximum number of turns
            assistant_message: The assistant's response message
        """
        if self._level > logging.INFO:
            return

        # Build panel content
        content = Text()

        # Add assistant content if present
        if assistant_message.content:
            text = assistant_message.content
            if isinstance(text, list):
                text = "\n".join(str(block) for block in text)
            # Truncate long content
            if len(text) > 500:
                text = text[:500] + "..."
            content.append(text, style="white")

        # Add tool calls if present
        if assistant_message.tool_calls:
            if assistant_message.content:
                content.append("\n\n")
            content.append("Tool Calls:\n", style="bold magenta")
            for tc in assistant_message.tool_calls:
                args_parsed = json.loads(tc.arguments)
                args_formatted = json.dumps(args_parsed, indent=2, ensure_ascii=False)
                args_preview = args_formatted[:1000] + "..." if len(args_formatted) > 1000 else args_formatted
                content.append(f"  🔧 {tc.name}", style="magenta")
                content.append(args_preview, style="dim")

        # Create and print panel with agent name in title
        title = f"[bold]AssistantMessage[/bold] │ {self.name} │ Turn {turn}/{max_turns}"
        panel = Panel(content, title=title, title_align="left", border_style="yellow", padding=(0, 1))

        if self.depth > 0:
            self._print_indented(panel, self.depth * SUBAGENT_INDENT_SPACES)
        else:
            console.print(panel)

    def user_message(self, user_message: UserMessage) -> None:
        """Log a user message in a panel.

        Args:
            user_message: The user's message
        """
        if self._level > logging.INFO:
            return

        # Build panel content
        content = Text()

        # Add user content
        if user_message.content:
            text = user_message.content
            if isinstance(text, list):
                text = "\n".join(str(block) for block in text)
            # Truncate long content
            if len(text) > 500:
                text = text[:500] + "..."
            content.append(text, style="white")

        # Create and print panel with agent name in title
        title = f"[bold]UserMessage[/bold] │ {self.name}"
        panel = Panel(content, title=title, title_align="left", border_style="blue", padding=(0, 1))

        if self.depth > 0:
            self._print_indented(panel, self.depth * SUBAGENT_INDENT_SPACES)
        else:
            console.print(panel)

    def task_message(self, task: str | list[Any]) -> None:
        """Log the initial task/prompt at the start of a run."""
        if self._level > logging.INFO:
            return

        # Convert list content to string
        if isinstance(task, list):
            task = "\n".join(str(block) for block in task)

        # Clean up whitespace from multi-line strings
        # Normalize each line by stripping leading/trailing whitespace and rejoining
        lines = [line.strip() for line in task.split("\n")]
        task = " ".join(line for line in lines if line)

        # Use "Sub Agent" prefix for nested agents
        prefix = "Sub Agent" if self.depth > 0 else "Agent"

        if self.depth > 0:
            indent = " " * (self.depth * SUBAGENT_INDENT_SPACES)
            console.print(f"{indent}[bold]{prefix} Task:[/bold]")
            console.print()
            for line in task.split("\n"):
                console.print(f"{indent}{line}")
        else:
            console.print(f"[bold]{prefix} Task:[/bold]")
            console.print()
            console.print(task)

        console.print()  # Add gap after task section

    def warnings_message(self, warnings: list[str]) -> None:
        """Display warnings at run start as simple text."""
        if self._level > logging.INFO or not warnings:
            return

        console.print("[bold orange1]Warnings[/bold orange1]")
        console.print()
        for warning in warnings:
            console.print(f"[orange1]⚠ {warning}[/orange1]")
            console.print()  # Add gap between warnings

    def tool_result(self, tool_message: ToolMessage) -> None:
        """Log a single tool execution result in a panel with XML syntax highlighting.

        Args:
            tool_message: The tool execution result
        """
        if self._level > logging.INFO:
            return

        tool_name = tool_message.name or "unknown"

        # Get result content
        result_text = tool_message.content
        if isinstance(result_text, list):
            result_text = "\n".join(str(block) for block in result_text)

        # Unescape HTML entities (e.g., &lt; -> <, &gt; -> >, &amp; -> &)
        result_text = html.unescape(result_text)

        # Truncate long results
        if len(result_text) > 1000:
            result_text = result_text[:1000] + "..."

        # Format as XML with syntax highlighting
        content = Syntax(result_text, "xml", theme="monokai", word_wrap=True)

        # Status indicator in title with agent name
        status = "✓" if tool_message.args_was_valid else "✗"
        status_style = "green" if tool_message.args_was_valid else "red"
        title = f"[{status_style}]{status}[/{status_style}] [bold]ToolResult[/bold] │ {self.name} │ [green]{tool_name}[/green]"

        panel = Panel(content, title=title, title_align="left", border_style="green", padding=(0, 1))

        if self.depth > 0:
            self._print_indented(panel, self.depth * SUBAGENT_INDENT_SPACES)
        else:
            console.print(panel)

    # -------------------------------------------------------------------------
    # Context Summarization Methods (AgentLoggerBase implementation)
    # -------------------------------------------------------------------------

    def context_summarization_start(self, pct_used: float, cutoff: float) -> None:
        """Log context window summarization starting in an orange panel.

        Args:
            pct_used: Percentage of context window currently used (0.0-1.0)
            cutoff: The threshold that triggered summarization (0.0-1.0)
        """
        # Build panel content
        content = Text()
        content.append("Context window limit reached\n\n", style="bold")
        content.append("Used: ", style="dim")
        content.append(f"{pct_used:.1%}", style="bold orange1")
        content.append("  │  ", style="dim")
        content.append("Threshold: ", style="dim")
        content.append(f"{cutoff:.1%}", style="bold")
        content.append("\n\n", style="dim")
        content.append("Summarizing conversation history...", style="italic")

        panel = Panel(
            content,
            title="[bold orange1]📝 Context Summarization[/]",
            title_align="left",
            border_style="orange1",
            padding=(0, 1),
        )

        if self.depth > 0:
            self._print_indented(panel, self.depth * SUBAGENT_INDENT_SPACES)
        else:
            console.print(panel)

    def context_summarization_complete(self, summary: str, bridge: str) -> None:
        """Log the completed context summarization with summary content.

        Args:
            summary: The generated summary of the conversation
            bridge: The bridge message that will be used to continue the conversation
        """
        # Truncate long summaries for display
        summary_display = summary
        if len(summary_display) > 800:
            summary_display = summary_display[:800] + "..."

        # Build panel content
        content = Text()
        content.append("Summary:\n", style="bold")
        content.append(summary_display, style="white")

        if self._level > logging.INFO:
            bridge_display = bridge
            if len(bridge_display) > 200:
                bridge_display = bridge_display[:200] + "..."
            content.append("\n\n")
            content.append("Bridge Message:\n", style="bold dim")
            content.append(bridge_display, style="dim italic")

        panel = Panel(
            content,
            title="[bold green]✓ Summary Generated[/]",
            title_align="left",
            border_style="green",
            padding=(0, 1),
        )

        if self.depth > 0:
            self._print_indented(panel, self.depth * SUBAGENT_INDENT_SPACES)
        else:
            console.print(panel)
