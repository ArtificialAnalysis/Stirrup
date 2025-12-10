"""LiteLLM-based LLM client for multi-provider support.

This client uses LiteLLM to provide a unified interface to multiple LLM providers
(OpenAI, Anthropic, Google, etc.) with automatic retries for transient failures.

Requires the litellm extra: `pip install stirrup[litellm]`
"""

import logging
from typing import Any

try:
    from litellm import acompletion
    from litellm.exceptions import APIConnectionError, RateLimitError, Timeout
except ImportError as e:
    raise ImportError(
        "Requires installation of the litellm extra. "
        "Install with: `uv pip install stirrup[litellm]` or `uv add stirrup[litellm]`"
    ) from e

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stirrup.clients.utils import to_openai_messages, to_openai_tools
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    LLMClient,
    Reasoning,
    TokenUsage,
    Tool,
    ToolCall,
)

__all__ = [
    "LiteLLMClient",
]

LOGGER = logging.getLogger(__name__)


class LiteLLMClient(LLMClient):
    """LiteLLM-based client supporting multiple LLM providers with unified interface.

    Includes automatic retries for transient failures and token usage tracking.
    """

    def __init__(
        self,
        model_slug: str,
        max_tokens: int,
        supports_audio_input: bool = False,
        reasoning_effort: str | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize LiteLLM client with model configuration and capabilities.

        Args:
            model_slug: Model identifier for LiteLLM (e.g., 'anthropic/claude-3-5-sonnet-20241022')
            max_tokens: Maximum context window size in tokens
            supports_audio_input: Whether the model supports audio inputs
            reasoning_effort: Reasoning effort level for extended thinking models (e.g., 'medium', 'high')
            kwargs: Additional arguments to pass to LiteLLM completion calls
        """
        self._model_slug = model_slug
        self._supports_video_input = False
        self._supports_audio_input = supports_audio_input
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._kwargs = kwargs or {}

    @property
    def max_tokens(self) -> int:
        """Maximum context window size in tokens."""
        return self._max_tokens

    @property
    def model_slug(self) -> str:
        """Model identifier used by LiteLLM."""
        return self._model_slug

    @retry(
        retry=retry_if_exception_type((Timeout, APIConnectionError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        """Generate assistant response with optional tool calls. Retries up to 3 times on timeout/connection errors."""
        r = await acompletion(
            model=self.model_slug,
            messages=to_openai_messages(messages),
            tools=to_openai_tools(tools) if tools else None,
            tool_choice="auto" if tools else None,
            max_tokens=self._max_tokens,
            **self._kwargs,
        )

        choice = r["choices"][0]

        if choice.finish_reason in ["max_tokens", "length"]:
            raise ContextOverflowError(
                f"Maximal context window tokens reached for model {self.model_slug}, resulting in finish reason: {choice.finish_reason}. Reduce agent.max_tokens and try again."
            )

        msg = choice["message"]

        reasoning: Reasoning | None = None
        if getattr(msg, "reasoning_content", None) is not None:
            reasoning = Reasoning(content=msg.reasoning_content)
        if getattr(msg, "thinking_blocks", None) is not None and len(msg.thinking_blocks) > 0:
            reasoning = Reasoning(
                signature=msg.thinking_blocks[0]["signature"], content=msg.thinking_blocks[0]["content"]
            )

        usage = r["usage"]

        calls = [
            ToolCall(
                tool_call_id=tc.get("id"),
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments", "") or "",
            )
            for tc in (msg.get("tool_calls") or [])
        ]

        input_tokens = usage.prompt_tokens
        reasoning_tokens = 0
        if usage.completion_tokens_details:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
        output_tokens = usage.completion_tokens - reasoning_tokens

        return AssistantMessage(
            reasoning=reasoning,
            content=msg.get("content") or "",
            tool_calls=calls,
            token_usage=TokenUsage(
                input=input_tokens,
                output=output_tokens,
                reasoning=reasoning_tokens,
            ),
        )
