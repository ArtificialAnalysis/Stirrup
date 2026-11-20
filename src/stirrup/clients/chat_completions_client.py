"""OpenAI SDK-based LLM client for chat completions.

This client uses the official OpenAI Python SDK directly, supporting both OpenAI's
API and any OpenAI-compatible endpoint via the `base_url` parameter (e.g., vLLM,
Ollama, Azure OpenAI, local models).

This is the default client for Stirrup.
"""

import logging
import os
from time import perf_counter
from typing import Any

from openai import AsyncOpenAI, BadRequestError
from openai.types.chat import ChatCompletion
from tenacity import AsyncRetrying, before_sleep_log, retry_if_exception, stop_after_attempt, wait_exponential

from stirrup.clients.utils import to_openai_messages, to_openai_tools, validate_token_budgets
from stirrup.core.exceptions import ContextOverflowError, EmptyResponseError, OutputTokenLimitError
from stirrup.core.models import (
    AssistantBlock,
    AssistantMessage,
    ChatMessage,
    LLMClient,
    ReasoningBlock,
    TextBlock,
    TokenUsage,
    Tool,
    ToolCall,
)

__all__ = [
    "ChatCompletionsClient",
]

LOGGER = logging.getLogger(__name__)

# Backoff between attempts when a reply carries no choices (see EmptyResponseError).
_EMPTY_RESPONSE_WAIT = wait_exponential(multiplier=1, min=1, max=10)

# Caps the provider error quoted in EmptyResponseError messages and retry warnings.
_MAX_ERROR_CHARS = 500

# 4xx codes that are not the client's fault and may succeed on a retry.
_RETRYABLE_CLIENT_ERROR_CODES = frozenset({408, 429})


def _is_retryable_empty_response(error: BaseException) -> bool:
    """Whether ``error`` is a reply without choices worth requesting again."""
    return isinstance(error, EmptyResponseError) and error.retryable


def _describe_provider_error(provider_error: object) -> str:
    """Render a gateway's ``error`` object, cut to ``_MAX_ERROR_CHARS`` characters."""
    text = str(provider_error)
    return text if len(text) <= _MAX_ERROR_CHARS else f"{text[:_MAX_ERROR_CHARS]}... [truncated]"


def _is_client_error(provider_error: object) -> bool:
    """Whether a gateway's ``error`` object carries a 4xx code a retry cannot fix."""
    code = provider_error.get("code") if isinstance(provider_error, dict) else None
    is_int = isinstance(code, int) and not isinstance(code, bool)
    return is_int and 400 <= code < 500 and code not in _RETRYABLE_CLIENT_ERROR_CODES


class ChatCompletionsClient(LLMClient):
    """OpenAI SDK-based client supporting OpenAI and OpenAI-compatible APIs.

    Uses the official OpenAI Python SDK directly for chat completions.
    Supports custom base_url for OpenAI-compatible providers (vLLM, Ollama,
    Azure OpenAI, local models, etc.).

    Delegates retries for transient failures to the OpenAI SDK and tracks token usage.
    Replies that succeed without any choices are retried by the client itself, up to
    the same ``max_retries``, unless their error carries a 4xx code other than 408 or 429.
    These retries wrap the SDK's own, so the two nest.

    Example:
        >>> # Standard OpenAI usage
        >>> client = ChatCompletionsClient(
        ...     model="gpt-5.6-luna",
        ...     max_tokens=8_192,
        ...     context_window_tokens=1_000_000,
        ... )
        >>>
        >>> # Custom OpenAI-compatible endpoint
        >>> client = ChatCompletionsClient(
        ...     model="llama-3.1-70b",
        ...     context_window_tokens=128_000,
        ...     base_url="http://localhost:8000/v1",
        ...     api_key="your-api-key",
        ... )
    """

    def __init__(
        self,
        model: str,
        max_tokens: int = 64_000,
        *,
        context_window_tokens: int,
        base_url: str | None = None,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OpenAI SDK client with model configuration.

        Args:
            model: Model identifier (e.g., 'gpt-5.6-luna', 'gpt-5.6-sol').
            max_tokens: Maximum number of tokens the provider may generate. Defaults to 64,000.
            context_window_tokens: Context capacity used to decide when Agent history
                should be summarized.
            base_url: API base URL. If None, uses OpenAI's standard URL.
                Use for OpenAI-compatible providers (e.g., 'http://localhost:8000/v1').
            api_key: API key for authentication. If None, reads from OPENROUTER_API_KEY
                environment variable.
            reasoning_effort: Reasoning effort level for extended thinking models
                (e.g., 'low', 'medium', 'high'). Only used with reasoning models.
            timeout: Request timeout in seconds. If None, uses OpenAI SDK default.
            max_retries: Number of retries for transient errors. Defaults to 2.
                The OpenAI SDK handles retries internally with exponential backoff.
                Also bounds the client's own retries of replies without choices, which
                wrap the SDK's retries, so the two nest.
            kwargs: Additional arguments passed to chat.completions.create().
                Values here override the base request parameters (model, messages,
                and the token cap) and bypass constructor validation; tool and
                reasoning parameters are always set by the client.

        Raises:
            ValueError: If ``context_window_tokens`` is not positive, or
                ``max_tokens`` exceeds it.
        """
        validate_token_budgets(max_tokens, context_window_tokens)

        self._model = model
        self._max_tokens = max_tokens
        self._context_window_tokens = context_window_tokens
        self._reasoning_effort = reasoning_effort
        self._kwargs = kwargs or {}
        self._max_retries = max_retries

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
        """Maximum number of tokens the provider may generate."""
        return self._max_tokens

    @property
    def context_window_tokens(self) -> int:
        """Context capacity used by agents for history summarization."""
        return self._context_window_tokens

    @property
    def model_slug(self) -> str:
        """Model identifier."""
        return self._model

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: dict[str, Tool],
    ) -> AssistantMessage:
        """Generate assistant response with optional tool calls.

        Args:
            messages: List of conversation messages.
            tools: Dictionary mapping tool names to Tool objects.

        Returns:
            AssistantMessage containing the model's response, any tool calls,
            and token usage statistics.

        Raises:
            ContextOverflowError: If the provider rejects the request because the input
                exceeds the model's context capacity.
            OutputTokenLimitError: If the provider exhausts ``max_tokens``.
            EmptyResponseError: If every attempt returned no choices, or one did with a
                4xx error code other than 408 or 429.
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

        # Add reasoning effort if configured (for reasoning models)
        if self._reasoning_effort:
            request_kwargs["reasoning_effort"] = self._reasoning_effort

        # Make API call, retrying replies without choices. Timing and token usage come only
        # from the successful attempt; usage of replies without choices is not counted.
        retrying = AsyncRetrying(
            retry=retry_if_exception(_is_retryable_empty_response),
            stop=stop_after_attempt(self._max_retries + 1),
            wait=_EMPTY_RESPONSE_WAIT,
            before_sleep=before_sleep_log(LOGGER, logging.WARNING),
            reraise=True,
        )
        response, request_start_time, request_end_time = await retrying(self._create_completion, request_kwargs)

        choice = response.choices[0]

        if choice.finish_reason in ("max_tokens", "length"):
            raise OutputTokenLimitError(
                model_slug=self.model_slug,
                max_tokens=self._max_tokens,
                provider_reason=choice.finish_reason,
            )

        msg = choice.message

        # Chat Completions is channel-shaped on the wire — no ordering to preserve —
        # so blocks are built in canonical channel order: reasoning → text → tool calls.
        blocks: list[AssistantBlock] = []

        # Parse reasoning content (for reasoning models with extended thinking)
        reasoning_content = getattr(msg, "reasoning_content", None)
        if reasoning_content:
            blocks.append(ReasoningBlock(content=reasoning_content))

        if msg.content:
            blocks.append(TextBlock(text=msg.content))

        # Parse tool calls
        blocks.extend(
            ToolCall.from_provider(
                provider_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "",
            )
            for tc in (msg.tool_calls or [])
        )

        # Parse token usage
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        # Handle reasoning tokens if available (for reasoning models)
        reasoning_tokens = 0
        if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

        answer_tokens = output_tokens - reasoning_tokens

        return AssistantMessage(
            blocks=blocks,
            token_usage=TokenUsage(
                input=input_tokens,
                answer=answer_tokens,
                reasoning=reasoning_tokens,
            ),
            request_start_time=request_start_time,
            request_end_time=request_end_time,
        )

    async def _create_completion(self, request_kwargs: dict[str, Any]) -> tuple[ChatCompletion, float, float]:
        """Make one chat completion request and return it with its start and end times.

        Raises:
            ContextOverflowError: If the provider rejects the input as too long.
            EmptyResponseError: If the reply has no choices, as OpenAI-compatible gateways
                such as OpenRouter return with HTTP 200 when an upstream provider fails.
        """
        request_start_time = perf_counter()
        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except BadRequestError as e:
            # Only OpenAI's own code is recognised. Compatible endpoints (vLLM, Ollama,
            # OpenRouter) that report a different code surface as a bad request instead of
            # being recovered; widening this deliberately trades that for false positives.
            if e.code != "context_length_exceeded":
                raise
            raise ContextOverflowError(str(e)) from e
        request_end_time = perf_counter()

        if not response.choices:
            # The SDK keeps fields it does not model, such as the gateway's error, in model_extra.
            provider_error = (response.model_extra or {}).get("error")
            detail = f": {_describe_provider_error(provider_error)}" if provider_error else ""
            raise EmptyResponseError(
                f"Model '{self.model_slug}' returned no choices{detail}",
                retryable=not _is_client_error(provider_error),
            )

        return response, request_start_time, request_end_time
