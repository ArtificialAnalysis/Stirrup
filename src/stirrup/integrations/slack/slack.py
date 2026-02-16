"""Stirrup Slack integration — run agents from Slack via @mentions.

Usage:
    @stirrup do xyz                              → default agent, default model
    @stirrup agent:data-analyst summarise this   → named agent
    @stirrup model:gpt-4o do xyz                 → default agent, override model
    @stirrup agent:data model:gpt-4o do xyz      → named agent + model override

Requires: pip install stirrup[slack]
"""

from __future__ import annotations

import asyncio
import copy
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Self

import httpx
from pydantic import BaseModel, ConfigDict

from stirrup.core.agent import Agent
from stirrup.core.models import (
    AssistantMessage,
    LLMClient,
    Tool,
    ToolMessage,
    ToolProvider,
    UserMessage,
    aggregate_metadata,
)
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider
from stirrup.tools.web import WebToolProvider
from stirrup.utils.logging import AgentLoggerBase

try:
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    from slack_bolt.async_app import AsyncApp
except ImportError as _err:
    raise ImportError(
        "slack-bolt is required for the Slack integration. Install with: pip install stirrup[slack]"
    ) from _err

logger = logging.getLogger(__name__)

__all__ = [
    "SlackAgentConfig",
    "SlackBot",
    "SlackBotConfig",
    "SlackLogger",
]

# Slack message character limit
_SLACK_MESSAGE_LIMIT = 40_000
_TRUNCATION_SUFFIX = "\n\n_...message truncated (exceeded Slack limit)..._"

# Default Dockerfile bundled with this package
_DEFAULT_DOCKERFILE = Path(__file__).parent / "Dockerfile"

# Regex for parsing agent: and model: prefixes from message text
_PREFIX_RE = re.compile(r"\b(agent|model):(\S+)")


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------


class SlackAgentConfig(BaseModel):
    """Configuration for a named agent identity."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    client: LLMClient
    tools: list[Tool | ToolProvider] | None = None
    system_prompt: str | None = None
    max_turns: int = 200
    skills_dir: str | None = None


class SlackBotConfig(BaseModel):
    """Top-level Slack bot configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    slack_bot_token: str
    slack_app_token: str
    default_agent: SlackAgentConfig
    named_agents: dict[str, SlackAgentConfig] = {}
    max_concurrent_runs: int = 5
    output_dir: str = "./slack_output"
    show_metadata: bool = True


# ---------------------------------------------------------------------------
# Minimal logger (logs agent events to standard Python logging)
# ---------------------------------------------------------------------------


class SlackLogger(AgentLoggerBase):
    """Minimal logger that logs agent events via standard Python logging.

    Outputs structured log lines to the terminal so you can monitor agent
    execution while the bot runs. Slack messaging is handled separately
    by the event handler.
    """

    def __init__(self) -> None:
        self.name: str = "agent"
        self.model: str | None = None
        self.max_turns: int | None = None
        self.depth: int = 0
        self.finish_params: BaseModel | None = None
        self.run_metadata: dict[str, list[Any]] | None = None
        self.output_dir: str | None = None

    def __enter__(self) -> Self:
        model_str = f" ({self.model})" if self.model else ""
        logger.info("[%s] Session started%s, max_turns=%s", self.name, model_str, self.max_turns)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if exc_type is not None:
            logger.error("[%s] Session ended with error: %s", self.name, exc_val)
        elif self.finish_params is None:
            logger.warning("[%s] Session ended without finishing (max turns reached?)", self.name)
        else:
            logger.info("[%s] Session completed successfully", self.name)

    def on_step(self, step: int, tool_calls: int = 0, input_tokens: int = 0, output_tokens: int = 0) -> None:
        logger.info(
            "[%s] Step %d/%s | %d tool calls | %d in / %d out tokens",
            self.name,
            step,
            self.max_turns or "?",
            tool_calls,
            input_tokens,
            output_tokens,
        )

    def assistant_message(self, turn: int, max_turns: int, assistant_message: AssistantMessage) -> None:
        tool_names = [tc.name for tc in assistant_message.tool_calls]
        content_preview = ""
        if assistant_message.content:
            text = (
                assistant_message.content
                if isinstance(assistant_message.content, str)
                else str(assistant_message.content)
            )
            content_preview = text[:200] + "..." if len(text) > 200 else text
        if tool_names:
            logger.info("[%s] Turn %d/%d | Tools: %s", self.name, turn, max_turns, ", ".join(tool_names))
        elif content_preview:
            logger.info("[%s] Turn %d/%d | %s", self.name, turn, max_turns, content_preview)

    def user_message(self, user_message: UserMessage) -> None:  # noqa: ARG002
        logger.debug("[%s] User message received", self.name)

    def task_message(self, task: str | list[Any]) -> None:
        task_str = task if isinstance(task, str) else str(task)
        preview = task_str[:200] + "..." if len(task_str) > 200 else task_str
        logger.info("[%s] Task: %s", self.name, preview)

    def tool_result(self, tool_message: ToolMessage) -> None:
        status = "ok" if tool_message.args_was_valid else "FAILED"
        logger.info("[%s] Tool %s → %s", self.name, tool_message.name, status)

    def context_summarization_start(self, pct_used: float, cutoff: float) -> None:
        logger.info(
            "[%s] Context summarization triggered (%.0f%% used, cutoff %.0f%%)", self.name, pct_used * 100, cutoff * 100
        )

    def context_summarization_complete(self, summary: str, bridge: str) -> None:  # noqa: ARG002
        logger.info("[%s] Context summarization complete", self.name)

    def debug(self, message: str, *args: object) -> None:
        logger.debug(message, *args)

    def info(self, message: str, *args: object) -> None:
        logger.info(message, *args)

    def warning(self, message: str, *args: object) -> None:
        logger.warning(message, *args)

    def error(self, message: str, *args: object) -> None:
        logger.error(message, *args)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _truncate_for_slack(text: str, limit: int = _SLACK_MESSAGE_LIMIT) -> str:
    """Truncate text to fit within Slack's message character limit."""
    if len(text) <= limit:
        return text
    max_content = limit - len(_TRUNCATION_SUFFIX)
    return text[:max_content] + _TRUNCATION_SUFFIX


def _format_finish_message(finish_params: BaseModel) -> str:
    """Format finish_params into a Slack-friendly message."""
    params_dict = finish_params.model_dump()
    reason = params_dict.get("reason", "Task completed.")
    paths = params_dict.get("paths", [])

    parts: list[str] = [reason]
    if paths:
        parts.append("\n:page_facing_up: *Output Files:*")
        parts.extend(f"  • `{path}`" for path in paths)

    return "\n".join(parts)


def _format_metadata_block(metadata: dict[str, list[Any]], agent_config: SlackAgentConfig) -> str | None:
    """Format run metadata as a compact Slack message."""
    if not metadata:
        return None

    aggregated: dict[str, Any] = aggregate_metadata(metadata, return_json_serializable=True)  # type: ignore[assignment]
    if not isinstance(aggregated, dict):
        return None

    parts: list[str] = [":bar_chart: *Run Metadata*"]

    # Token usage
    token_usage = aggregated.get("token_usage")
    if token_usage and isinstance(token_usage, list) and token_usage:
        usage = token_usage[0] if isinstance(token_usage[0], dict) else {}
        input_t = usage.get("input", 0)
        answer_t = usage.get("answer", 0)
        reasoning_t = usage.get("reasoning", 0)
        total = input_t + answer_t + reasoning_t
        parts.append(f"  Tokens: {total:,} total ({input_t:,} in, {answer_t:,} out)")
        if reasoning_t > 0:
            parts.append(f"  Reasoning tokens: {reasoning_t:,}")

    # Model info
    parts.append(f"  Model: `{agent_config.client.model_slug}`")

    if len(parts) <= 1:
        return None

    return "\n".join(parts)


def _format_error_message(error: Exception) -> str:
    """Format an exception as a user-friendly Slack error message."""
    error_type = type(error).__name__
    error_msg = str(error)
    if len(error_msg) > 1000:
        error_msg = error_msg[:1000] + "..."

    return f":x: *Agent Error*\n```\n{error_type}: {error_msg}\n```\n_Please try again or contact an administrator._"


# ---------------------------------------------------------------------------
# SlackBot
# ---------------------------------------------------------------------------


class SlackBot:
    """Stirrup Slack bot that routes @mentions to configured agents.

    Usage:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient
        from stirrup.integrations.slack import SlackBot, SlackBotConfig, SlackAgentConfig

        client = ChatCompletionsClient(model="google/gemini-3-flash-preview", base_url="https://openrouter.ai/api/v1")
        config = SlackBotConfig(
            slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
            slack_app_token=os.environ["SLACK_APP_TOKEN"],
            default_agent=SlackAgentConfig(name="assistant", client=client),
        )
        bot = SlackBot(config)
        await bot.start()
    """

    def __init__(self, config: SlackBotConfig) -> None:
        self._config = config
        self._app = AsyncApp(token=config.slack_bot_token)
        self._semaphore = asyncio.Semaphore(config.max_concurrent_runs)
        self._register_handlers()

    # -- Message parsing ------------------------------------------------------

    def parse_message(self, text: str) -> tuple[SlackAgentConfig, str, str | None]:
        """Parse a Slack message to extract agent config, prompt, and optional model override.

        Args:
            text: Raw message text from Slack (with <@BOT_ID> mention).

        Returns:
            Tuple of (agent_config, cleaned_prompt, model_override_or_None).
        """
        # Strip all <@USER_ID> mentions
        cleaned = re.sub(r"<@\w+>\s*", "", text).strip()

        # Extract agent: and model: prefixes
        agent_name: str | None = None
        model_override: str | None = None

        for match in _PREFIX_RE.finditer(cleaned):
            key, value = match.group(1), match.group(2)
            if key == "agent":
                agent_name = value
            elif key == "model":
                model_override = value

        # Remove the prefix tokens from the prompt
        prompt = _PREFIX_RE.sub("", cleaned).strip()

        # Resolve agent config
        if agent_name and agent_name in self._config.named_agents:
            agent_config = self._config.named_agents[agent_name]
        else:
            agent_config = self._config.default_agent

        return agent_config, prompt, model_override

    # -- Agent building -------------------------------------------------------

    def build_agent(self, config: SlackAgentConfig, model_override: str | None = None) -> Agent:
        """Build a Stirrup Agent from a SlackAgentConfig.

        Args:
            config: Agent configuration.
            model_override: Optional model name to override the client's default model.

        Returns:
            Configured Agent instance.
        """
        client = config.client
        if model_override is not None:
            # Shallow copy the client and swap the model name.
            # This preserves base_url, api_key, etc. from the original config.
            client = copy.copy(client)
            client._model = model_override  # type: ignore[attr-defined]  # noqa: SLF001

        # Deep copy tools so each concurrent run gets its own ToolProvider instances
        # (ToolProviders have lifecycle state that can't be shared across sessions).
        tools: list[Tool | ToolProvider] = (
            copy.deepcopy(config.tools)
            if config.tools is not None
            else [
                DockerCodeExecToolProvider.from_dockerfile(_DEFAULT_DOCKERFILE),
                WebToolProvider(),
            ]
        )

        return Agent(
            client=client,
            name=config.name,
            tools=tools,
            system_prompt=config.system_prompt,
            max_turns=config.max_turns,
            logger=SlackLogger(),
        )

    # -- Slack event handlers -------------------------------------------------

    def _register_handlers(self) -> None:
        """Register Slack event handlers on the AsyncApp."""

        @self._app.event("app_mention")
        async def handle_mention(event: dict, say: Any, client: Any) -> None:  # noqa: ANN401
            channel = event["channel"]
            thread_ts = event.get("thread_ts", event["ts"])
            text = event.get("text", "")
            files = event.get("files", [])

            agent_config, prompt, model_override = self.parse_message(text)

            if not prompt and not files:
                await say(
                    text="Please provide a task after mentioning me. Example: `@stirrup analyze this data`",
                    thread_ts=thread_ts,
                )
                return

            # Post initial status in thread
            status_msg = await say(
                text=(
                    f":hourglass_flowing_sand: Working on your request...\n"
                    f"_Agent: {agent_config.name} | Model: {model_override or agent_config.client.model_slug}_"
                ),
                thread_ts=thread_ts,
            )
            status_ts = status_msg["ts"]

            # Dispatch agent run as background task.
            # Store reference to prevent garbage collection (RUF006).
            task = asyncio.create_task(
                self._run_agent_task(
                    slack_client=client,
                    say=say,
                    agent_config=agent_config,
                    model_override=model_override,
                    prompt=prompt,
                    files=files,
                    channel=channel,
                    thread_ts=thread_ts,
                    status_ts=status_ts,
                )
            )
            task.add_done_callback(lambda t: t.result() if not t.cancelled() and t.exception() is None else None)

    async def _run_agent_task(
        self,
        *,
        slack_client: Any,  # noqa: ANN401
        say: Any,  # noqa: ANN401
        agent_config: SlackAgentConfig,
        model_override: str | None,
        prompt: str,
        files: list[dict],
        channel: str,
        thread_ts: str,
        status_ts: str,
    ) -> None:
        """Execute an agent run as a background task with concurrency control."""
        temp_dir: str | None = None
        try:
            async with self._semaphore:
                # Download attached files
                input_files: list[Path] = []
                if files:
                    temp_dir = tempfile.mkdtemp(prefix="stirrup_slack_")
                    input_files = await _download_slack_files(files, Path(temp_dir), self._config.slack_bot_token)

                # Create per-run output directory
                output_dir = Path(self._config.output_dir) / f"slack_{thread_ts.replace('.', '_')}"
                output_dir.mkdir(parents=True, exist_ok=True)

                # Build and run the agent
                agent = self.build_agent(agent_config, model_override)

                async with agent.session(
                    output_dir=str(output_dir),
                    input_files=[str(f) for f in input_files] if input_files else None,
                    skills_dir=agent_config.skills_dir,
                    cache_on_interrupt=False,
                ) as session:
                    finish_params, _history, metadata = await session.run(prompt)

                # Post results
                if finish_params is not None:
                    response_text = _truncate_for_slack(_format_finish_message(finish_params))
                    await say(text=response_text, thread_ts=thread_ts)

                    # Upload output files
                    paths = getattr(finish_params, "paths", None)
                    if paths:
                        await _upload_output_files(slack_client, channel, thread_ts, output_dir, paths)

                    # Post metadata
                    if self._config.show_metadata:
                        meta_text = _format_metadata_block(metadata, agent_config)
                        if meta_text:
                            await say(text=meta_text, thread_ts=thread_ts)
                else:
                    await say(
                        text=":warning: Agent could not complete the task within the turn limit.",
                        thread_ts=thread_ts,
                    )

                # Update status message
                await slack_client.chat_update(
                    channel=channel,
                    ts=status_ts,
                    text=f":white_check_mark: Completed | Agent: {agent_config.name}",
                )

        except Exception as exc:
            logger.exception("Agent run failed for thread %s", thread_ts)
            error_msg = _format_error_message(exc)
            try:
                await say(text=error_msg, thread_ts=thread_ts)
                await slack_client.chat_update(
                    channel=channel,
                    ts=status_ts,
                    text=f":x: Failed | Agent: {agent_config.name}",
                )
            except Exception:
                logger.exception("Failed to post error message to Slack")
        finally:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)

    # -- Start methods --------------------------------------------------------

    async def start(self) -> None:
        """Start the bot using Socket Mode (no public URL needed)."""
        handler = AsyncSocketModeHandler(self._app, self._config.slack_app_token)
        logger.info("Starting Stirrup Slack bot (Socket Mode)...")
        await handler.start_async()

    async def start_http(self, port: int = 3000) -> None:
        """Start the bot using HTTP mode (for production with a public URL)."""
        logger.info("Starting Stirrup Slack bot (HTTP mode, port %d)...", port)
        await self._app.start_async(port=port)


# ---------------------------------------------------------------------------
# File handling helpers
# ---------------------------------------------------------------------------


async def _download_slack_files(
    files: list[dict],
    dest_dir: Path,
    bot_token: str,
) -> list[Path]:
    """Download Slack-attached files to a local directory."""
    downloaded: list[Path] = []
    async with httpx.AsyncClient() as http_client:
        for file_info in files:
            url = file_info.get("url_private_download") or file_info.get("url_private")
            if not url:
                continue
            filename = file_info.get("name", "unknown_file")
            dest_path = dest_dir / filename

            response = await http_client.get(
                url,
                headers={"Authorization": f"Bearer {bot_token}"},
                follow_redirects=True,
            )
            if response.status_code == 200:
                dest_path.write_bytes(response.content)
                downloaded.append(dest_path)
                logger.info("Downloaded Slack file: %s -> %s", filename, dest_path)
            else:
                logger.warning("Failed to download %s: HTTP %d", filename, response.status_code)

    return downloaded


async def _upload_output_files(
    slack_client: Any,  # noqa: ANN401
    channel: str,
    thread_ts: str,
    output_dir: Path,
    paths: list[str],
) -> None:
    """Upload output files back to the Slack thread."""
    for path_str in paths:
        file_path = output_dir / path_str
        if file_path.exists() and file_path.is_file():
            try:
                await slack_client.files_upload_v2(
                    channel=channel,
                    thread_ts=thread_ts,
                    file=str(file_path),
                    filename=file_path.name,
                    initial_comment=f":page_facing_up: `{path_str}`",
                )
                logger.info("Uploaded file to Slack: %s", path_str)
            except Exception:
                logger.exception("Failed to upload %s to Slack", path_str)
        else:
            logger.warning("Output file not found: %s", file_path)
