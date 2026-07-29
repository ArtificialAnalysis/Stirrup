"""LiteLLM-based LLM client for multi-provider support.

This client uses LiteLLM to provide a unified interface to multiple LLM providers
(OpenAI, Anthropic, Google, etc.) with automatic retries for transient failures.

Requires the litellm extra: `pip install stirrup[litellm]`
"""

import logging
import warnings
from time import perf_counter
from typing import Any, Literal

try:
    from litellm import acompletion
    from litellm.exceptions import APIConnectionError, ContextWindowExceededError, RateLimitError, Timeout
except ImportError as e:
    raise ImportError(
        "Requires installation of the litellm extra. "
        "Install with: `uv pip install stirrup[litellm]` or `uv add stirrup[litellm]`"
    ) from e

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stirrup.clients.utils import to_openai_messages, to_openai_tools
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError
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

type ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "default"]


class LiteLLMClient(LLMClient):
    """LiteLLM-based client supporting multiple LLM providers with unified interface.

    Includes automatic retries for transient failures and token usage tracking.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 64_000,
        *,
        context_window_tokens: int | None = None,
        model_slug: str | None = None,
        api_key: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize LiteLLM client with model configuration and capabilities.

        Args:
            model: Model identifier for LiteLLM (e.g., 'anthropic/claude-3-5-sonnet-20241022')
            max_tokens: Maximum number of tokens the provider may generate
            context_window_tokens: Context capacity used to decide when Agent history
                should be summarized. Defaults to ``max_tokens`` when omitted.
            model_slug: Deprecated. Use model instead.
            reasoning_effort: Reasoning effort level for extended thinking models (e.g., 'medium', 'high')
            kwargs: Additional arguments to pass to LiteLLM completion calls

        Raises:
            ValueError: If no model is provided or ``context_window_tokens`` is not positive.
        """
        if model_slug is not None:
            warnings.warn(
                "The 'model_slug' parameter is deprecated. Use 'model' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if model is None:
                model = model_slug
        if model is None:
            raise ValueError("model is required")
        resolved_context_window_tokens = max_tokens if context_window_tokens is None else context_window_tokens
        if resolved_context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be positive")

        self._model = model
        self._max_tokens = max_tokens
        self._context_window_tokens = resolved_context_window_tokens
        self._reasoning_effort: ReasoningEffort | None = reasoning_effort
        self._api_key = api_key
        self._kwargs = kwargs or {}

    @property
    def max_tokens(self) -> int:
        """Maximum number of tokens the provider may generate."""
        return self._max_tokens

    @property
    def context_window_tokens(self) -> int:
        """Context capacity used by agents for history summarization."""
        return self._context_window_tokens

    @property
    def model_slug(self) -> str:
        """Model identifier used by LiteLLM."""
        return self._model

    @retry(
        retry=retry_if_exception_type((Timeout, APIConnectionError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        """Generate assistant response with optional tool calls. Retries up to 3 times on timeout/connection errors."""
        request_start_time = perf_counter()
        try:
            r = await acompletion(
                model=self.model_slug,
                messages=to_openai_messages(messages),
                tools=to_openai_tools(tools) if tools else None,
                tool_choice="auto" if tools else None,
                max_tokens=self._max_tokens,
                reasoning_effort=self._reasoning_effort,
                api_key=self._api_key,
                **self._kwargs,
            )
        except ContextWindowExceededError as e:
            raise ContextOverflowError(str(e)) from e
        request_end_time = perf_counter()

        choice = r["choices"][0]

        if choice.finish_reason in ("max_tokens", "length"):
            raise OutputTokenLimitError(
                model_slug=self.model_slug,
                max_tokens=self._max_tokens,
                provider_reason=choice.finish_reason,
            )

        msg = choice["message"]
        reasoning: Reasoning | None = None
        if getattr(msg, "reasoning_content", None) is not None:
            reasoning = Reasoning(content=msg.reasoning_content)
        if getattr(msg, "thinking_blocks", None) is not None and len(msg.thinking_blocks) > 0:
            if len(msg.thinking_blocks) > 1:
                raise ValueError("Found multiple thinking blocks in the response")

            signature = msg.thinking_blocks[0].get("thinking_signature", None)
            content = msg.thinking_blocks[0].get("thinking", None)

            if signature is None and content is None:
                raise ValueError("Signature and content not found in the thinking block response")

            reasoning = Reasoning(signature=signature, content=content)

        usage = r["usage"]

        calls = [
            ToolCall(
                tool_call_id=tc.get("id"),
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments", "") or "",
                signature=tc.get("provider_specific_fields", {}).get("thought_signature", None),
            )
            for tc in (msg.get("tool_calls") or [])
        ]

        input_tokens = usage.prompt_tokens
        reasoning_tokens = 0
        if usage.completion_tokens_details:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
        output_tokens = usage.completion_tokens
        answer_tokens = output_tokens - reasoning_tokens

        return AssistantMessage(
            reasoning=reasoning,
            content=msg.get("content") or "",
            tool_calls=calls,
            token_usage=TokenUsage(
                input=input_tokens,
                answer=answer_tokens,
                reasoning=reasoning_tokens,
            ),
            request_start_time=request_start_time,
            request_end_time=request_end_time,
        )
