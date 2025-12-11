"""OpenAI SDK-based LLM client for chat completions.

This client uses the official OpenAI Python SDK directly, supporting both OpenAI's
API and any OpenAI-compatible endpoint via the `base_url` parameter (e.g., vLLM,
Ollama, Azure OpenAI, local models).

This is the default client for Stirrup.
"""

import logging
import os
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
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
    "ChatCompletionsClient",
]

LOGGER = logging.getLogger(__name__)


class ChatCompletionsClient(LLMClient):
    """OpenAI SDK-based client supporting OpenAI and OpenAI-compatible APIs.

    Uses the official OpenAI Python SDK directly for chat completions.
    Supports custom base_url for OpenAI-compatible providers (vLLM, Ollama,
    Azure OpenAI, local models, etc.).

    Includes automatic retries for transient failures and token usage tracking.

    Example:
        >>> # Standard OpenAI usage
        >>> client = ChatCompletionsClient(model="gpt-4o", max_tokens=128_000)
        >>>
        >>> # Custom OpenAI-compatible endpoint
        >>> client = ChatCompletionsClient(
        ...     model="llama-3.1-70b",
        ...     base_url="http://localhost:8000/v1",
        ...     api_key="your-api-key",
        ... )
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 64_000,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OpenAI SDK client with model configuration.

        Args:
            model: Model identifier (e.g., 'gpt-5', 'gpt-4o', 'o1-preview').
            max_tokens: Maximum context window size in tokens. Defaults to 64,000.
            base_url: API base URL. If None, uses OpenAI's standard URL.
                Use for OpenAI-compatible providers (e.g., 'http://localhost:8000/v1').
            api_key: API key for authentication. If None, reads from OPENROUTER_API_KEY
                environment variable.
            reasoning_effort: Reasoning effort level for extended thinking models
                (e.g., 'low', 'medium', 'high'). Only used with o1/o3 style models.
            timeout: Request timeout in seconds. If None, uses OpenAI SDK default.
            max_retries: Number of retries for transient errors. Defaults to 2.
                The OpenAI SDK handles retries internally with exponential backoff.
            kwargs: Additional arguments passed to chat.completions.create().
        """
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._kwargs = kwargs or {}

        # Initialize AsyncOpenAI client
        # Read from OPENROUTER_API_KEY if no api_key provided
        resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self._client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def max_tokens(self) -> int:
        """Maximum context window size in tokens."""
        return self._max_tokens

    @property
    def model_slug(self) -> str:
        """Model identifier."""
        return self._model

    @retry(
        retry=retry_if_exception_type(
            (
                APIConnectionError,
                APITimeoutError,
                RateLimitError,
                InternalServerError,
            )
        ),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(
        self,
        messages: list[ChatMessage],
        tools: dict[str, Tool],
    ) -> AssistantMessage:
        """Generate assistant response with optional tool calls.

        Retries up to 3 times on transient errors (connection, timeout, rate limit,
        internal server errors) with exponential backoff.

        Args:
            messages: List of conversation messages.
            tools: Dictionary mapping tool names to Tool objects.

        Returns:
            AssistantMessage containing the model's response, any tool calls,
            and token usage statistics.

        Raises:
            ContextOverflowError: If the context window is exceeded.
        """
        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": to_openai_messages(messages),
            "max_completion_tokens": self._max_tokens,
            **self._kwargs,
        }

        # Add tools if provided
        if tools:
            request_kwargs["tools"] = to_openai_tools(tools)
            request_kwargs["tool_choice"] = "auto"

        # Add reasoning effort if configured (for o1/o3 models)
        if self._reasoning_effort:
            request_kwargs["reasoning_effort"] = self._reasoning_effort

        # Make API call
        response = await self._client.chat.completions.create(**request_kwargs)

        choice = response.choices[0]

        # Check for context overflow
        if choice.finish_reason in ("max_tokens", "length"):
            raise ContextOverflowError(
                f"Maximal context window tokens reached for model {self.model_slug}, "
                f"resulting in finish reason: {choice.finish_reason}. "
                "Reduce agent.max_tokens and try again."
            )

        msg = choice.message

        # Parse reasoning content (for o1/o3 models with extended thinking)
        reasoning: Reasoning | None = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning = Reasoning(content=msg.reasoning_content)

        # Parse tool calls
        tool_calls = [
            ToolCall(
                tool_call_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "",
            )
            for tc in (msg.tool_calls or [])
        ]

        # Parse token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Handle reasoning tokens if available (for o1/o3 models)
        reasoning_tokens = 0
        if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
            output_tokens = output_tokens - reasoning_tokens

        return AssistantMessage(
            reasoning=reasoning,
            content=msg.content or "",
            tool_calls=tool_calls,
            token_usage=TokenUsage(
                input=input_tokens,
                output=output_tokens,
                reasoning=reasoning_tokens,
            ),
        )
