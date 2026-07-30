"""OpenAI SDK-based LLM client for the Responses API.

This client uses the official OpenAI Python SDK's responses.create() method,
supporting both OpenAI's API and any OpenAI-compatible endpoint that implements
the Responses API via the `base_url` parameter.
"""

import logging
from time import perf_counter
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stirrup.clients.utils import _resolve_openai_endpoint_and_api_key
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (
    AssistantMessage,
    AudioContentBlock,
    ChatMessage,
    Content,
    EmptyParams,
    ImageContentBlock,
    LLMClient,
    Reasoning,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
    VideoContentBlock,
)

__all__ = [
    "OpenResponsesClient",
]

LOGGER = logging.getLogger(__name__)


def _content_to_open_responses_input(content: Content) -> list[dict[str, Any]]:
    """Convert Content blocks to OpenResponses input content format.

    Uses input_text for text content (vs output_text for responses).
    """
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]

    out: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            out.append({"type": "input_text", "text": block})
        elif isinstance(block, ImageContentBlock):
            out.append({"type": "input_image", "image_url": block.to_base64_url()})
        elif isinstance(block, AudioContentBlock):
            out.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": block.to_base64_url().split(",")[1],
                        "format": block.extension,
                    },
                }
            )
        elif isinstance(block, VideoContentBlock):
            out.append({"type": "input_file", "file_data": block.to_base64_url()})
        else:
            raise NotImplementedError(f"Unsupported content block: {type(block)}")
    return out


def _content_to_open_responses_output(content: Content) -> list[dict[str, Any]]:
    """Convert Content blocks to OpenResponses output content format.

    Uses output_text for assistant message content.
    """
    if isinstance(content, str):
        return [{"type": "output_text", "text": content}]

    out: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block, str):
            out.append({"type": "output_text", "text": block})
        else:
            raise NotImplementedError(f"Unsupported output content block: {type(block)}")
    return out


def _to_open_responses_tools(tools: dict[str, Tool]) -> list[dict[str, Any]]:
    """Convert Tool objects to OpenResponses function format.

    OpenResponses API expects tools with name/description/parameters at top level,
    not nested under a 'function' key like Chat Completions API.

    Args:
        tools: Dictionary mapping tool names to Tool objects.

    Returns:
        List of tool definitions in OpenResponses format.
    """
    out: list[dict[str, Any]] = []
    for t in tools.values():
        tool_def: dict[str, Any] = {
            "type": "function",
            "name": t.name,
            "description": t.description,
        }
        if t.parameters is not EmptyParams:
            tool_def["parameters"] = t.parameters.model_json_schema()
        out.append(tool_def)
    return out


def _to_open_responses_input(
    msgs: list[ChatMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert ChatMessage list to OpenResponses (instructions, input) tuple.

    SystemMessage content is extracted as the instructions parameter.
    Other messages are converted to input items.

    Returns:
        Tuple of (instructions, input_items) where instructions is the system
        message content (or None) and input_items is the list of input items.
    """
    instructions: str | None = None
    input_items: list[dict[str, Any]] = []

    for m in msgs:
        if isinstance(m, SystemMessage):
            # Extract system message as instructions
            if isinstance(m.content, str):
                instructions = m.content
            else:
                # Join text content blocks for instructions
                instructions = "\n".join(block if isinstance(block, str) else "" for block in m.content)
        elif isinstance(m, UserMessage):
            input_items.append(
                {
                    "role": "user",
                    "content": _content_to_open_responses_input(m.content),
                }
            )
        elif isinstance(m, AssistantMessage):
            # For assistant messages, we need to add them as response output items
            # First add any text content as a message item
            content_str = (
                m.content
                if isinstance(m.content, str)
                else "\n".join(block if isinstance(block, str) else "" for block in m.content)
            )
            if content_str:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content_str}],
                    }
                )

            # Add tool calls as separate function_call items
            input_items.extend(
                {
                    "type": "function_call",
                    "call_id": tc.tool_call_id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in m.tool_calls
            )
        elif isinstance(m, ToolMessage):
            # Tool results are function_call_output items
            content_str = m.content if isinstance(m.content, str) else str(m.content)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": content_str,
                }
            )
        else:
            raise NotImplementedError(f"Unsupported message type: {type(m)}")

    return instructions, input_items


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:  # noqa: ANN401
    """Get attribute from object or dict, with fallback default."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_response_output(
    output: list[Any],
) -> tuple[str, list[ToolCall], Reasoning | None]:
    """Parse response output items into content, tool_calls, and reasoning.

    Args:
        output: List of output items from the response.

    Returns:
        Tuple of (content_text, tool_calls, reasoning).
    """
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    reasoning: Reasoning | None = None

    for item in output:
        item_type = _get_attr(item, "type")

        if item_type == "message":
            # Extract text content from message
            msg_content = _get_attr(item, "content", [])
            for content_item in msg_content:
                content_type = _get_attr(content_item, "type")
                if content_type == "output_text":
                    text = _get_attr(content_item, "text", "")
                    content_parts.append(text)

        elif item_type == "function_call":
            call_id = _get_attr(item, "call_id")
            name = _get_attr(item, "name")
            arguments = _get_attr(item, "arguments", "")
            tool_calls.append(
                ToolCall(
                    tool_call_id=call_id,
                    name=name,
                    arguments=arguments,
                )
            )

        elif item_type == "reasoning":
            # Extract reasoning/thinking content - try multiple possible attribute names
            # summary can be a list of Summary objects with .text attribute
            summary = _get_attr(item, "summary")
            if summary:
                if isinstance(summary, list):
                    # Extract text from Summary objects
                    thinking = "\n".join(_get_attr(s, "text", "") for s in summary if _get_attr(s, "text"))
                else:
                    thinking = str(summary)
            else:
                thinking = _get_attr(item, "thinking") or ""

            if thinking:
                reasoning = Reasoning(content=thinking)

    return "\n".join(content_parts), tool_calls, reasoning


class OpenResponsesClient(LLMClient):
    """OpenAI SDK-based client using the Responses API.

    Uses the official OpenAI Python SDK's responses.create() method.
    Supports custom base_url for OpenAI-compatible providers that implement
    the Responses API.

    Includes automatic retries for transient failures and token usage tracking.

    Example:
        >>> # Standard OpenAI usage
        >>> client = OpenResponsesClient(model="gpt-4o", max_tokens=128_000)
        >>>
        >>> # Custom OpenAI-compatible endpoint
        >>> client = OpenResponsesClient(
        ...     model="gpt-4o",
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
        instructions: str | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OpenAI SDK client with model configuration for Responses API.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'o1-preview').
            max_tokens: Maximum output tokens. Defaults to 64,000.
            base_url: API base URL. If None, reads ``OPENAI_BASE_URL`` before
                using OpenAI's standard URL. Use for OpenAI-compatible providers.
            api_key: API key for authentication. Required for every endpoint except
                OpenAI's own ``https://api.openai.com``, where the OpenAI SDK reads
                ``OPENAI_API_KEY`` itself. Stirrup never reads a credential from the
                environment.
            reasoning_effort: Reasoning effort level for extended thinking models
                (e.g., 'low', 'medium', 'high'). Only used with o1/o3 style models.
            timeout: Request timeout in seconds. If None, uses OpenAI SDK default.
            max_retries: Number of retries for transient errors. Defaults to 2.
            instructions: Default system-level instructions. Can be overridden by
                SystemMessage in the messages list.
            kwargs: Additional arguments passed to responses.create().

        Raises:
            openai.OpenAIError: If the effective base URL is not an absolute HTTP(S) URL,
                includes userinfo, a query or a fragment, or is any endpoint other than
                OpenAI's own HTTPS one while ``api_key`` is omitted. The OpenAI SDK raises
                the same error type when that endpoint is used and ``OPENAI_API_KEY`` is
                unset. A base URL httpx cannot parse at all (a non-numeric port, say) still
                surfaces as ``httpx.InvalidURL``, since httpx parses it before Stirrup
                inspects it.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._default_instructions = instructions
        self._kwargs = kwargs or {}

        endpoint, resolved_api_key = _resolve_openai_endpoint_and_api_key(base_url, api_key)

        # The SDK appends /responses itself. Strip it from the encoded path, because
        # ``endpoint.path`` is percent-decoded and writing it back would turn a %2F inside
        # a segment into a real separator and re-route a path-routing gateway. ``raw_path``
        # would also carry a query, but the resolver rejects a base URL that has one.
        raw_path = endpoint.raw_path.rstrip(b"/")
        if raw_path.endswith(b"/responses"):
            endpoint = endpoint.copy_with(raw_path=raw_path.removesuffix(b"/responses"))

        self._client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=endpoint,
            timeout=timeout,
            max_retries=max_retries,
        )

    @property
    def max_tokens(self) -> int:
        """Maximum output tokens."""
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
        """Generate assistant response with optional tool calls using Responses API.

        Retries up to 3 times on transient errors (connection, timeout, rate limit,
        internal server errors) with exponential backoff.

        Args:
            messages: List of conversation messages.
            tools: Dictionary mapping tool names to Tool objects.

        Returns:
            AssistantMessage containing the model's response, any tool calls,
            and token usage statistics.

        Raises:
            ContextOverflowError: If the response is incomplete due to token limits.
        """
        # Convert messages to OpenResponses format
        instructions, input_items = _to_open_responses_input(messages)

        # Use provided instructions or fall back to default
        final_instructions = instructions or self._default_instructions

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_items,
            "max_output_tokens": self._max_tokens,
            **self._kwargs,
        }

        # Add instructions if present
        if final_instructions:
            request_kwargs["instructions"] = final_instructions

        # Add tools if provided
        if tools:
            request_kwargs["tools"] = _to_open_responses_tools(tools)
            request_kwargs["tool_choice"] = "auto"

        # Add reasoning effort if configured (for o1/o3 models)
        if self._reasoning_effort:
            request_kwargs["reasoning"] = {"effort": self._reasoning_effort}

        # Make API call
        request_start_time = perf_counter()
        response = await self._client.responses.create(**request_kwargs)
        request_end_time = perf_counter()

        # Check for incomplete response (context overflow)
        if response.status == "incomplete":
            stop_reason = getattr(response, "incomplete_details", None)
            raise ContextOverflowError(
                f"Response incomplete for model {self.model_slug}: {stop_reason}. "
                "Reduce max_tokens or message length and try again."
            )

        # Parse response output
        content, tool_calls, reasoning = _parse_response_output(response.output)

        # Parse token usage
        usage = response.usage
        input_tokens = usage.input_tokens if usage else 0
        output_tokens = usage.output_tokens if usage else 0

        # Handle reasoning tokens if available
        reasoning_tokens = 0
        if usage and hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
            reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0

        answer_tokens = output_tokens - reasoning_tokens

        return AssistantMessage(
            reasoning=reasoning,
            content=content,
            tool_calls=tool_calls,
            token_usage=TokenUsage(
                input=input_tokens,
                answer=answer_tokens,
                reasoning=reasoning_tokens,
            ),
            request_start_time=request_start_time,
            request_end_time=request_end_time,
        )
