"""OpenAI SDK-based LLM client for the Responses API.

This client uses the official OpenAI Python SDK's responses.create() method,
supporting both OpenAI's API and any OpenAI-compatible endpoint that implements
the Responses API via the `base_url` parameter.
"""

import logging
import os
from collections.abc import Sequence
from time import perf_counter
from typing import Any, assert_never

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from openai.types.responses import Response
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (
    AssistantBlock,
    AssistantMessage,
    AudioContentBlock,
    ChatMessage,
    Content,
    EmptyParams,
    EncryptedReasoningBlock,
    ImageContentBlock,
    LLMClient,
    OpaqueBlock,
    ReasoningBlock,
    ReasoningRefBlock,
    RedactedReasoningBlock,
    SignedReasoningBlock,
    SystemMessage,
    TextBlock,
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

logger = logging.getLogger(__name__)

_OWNED_REQUEST_KEYS = frozenset(
    {
        "background",
        "conversation",
        "input",
        "instructions",
        "max_output_tokens",
        "model",
        "previous_response_id",
        "store",
        "stream",
    }
)


def _is_missing_previous_response(error: APIStatusError) -> bool:
    """Match only the provider error that identifies an unavailable continuation."""
    return error.code == "previous_response_not_found" and error.param == "previous_response_id"


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
    msgs: Sequence[ChatMessage],
    *,
    allow_reference_reasoning: bool = True,
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
            # Assistant turns replay as response output items, one item per block,
            # in the model's true emission order (message / function_call / reasoning
            # interleaved exactly as captured). Channel-synthesized (legacy) messages
            # carry blocks in channel order, so they replay in the old
            # message-then-calls order automatically.
            for block in m.blocks:
                match block:
                    case TextBlock(text=text, signature=None):
                        input_items.append(
                            {
                                "type": "message",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": text}],
                            }
                        )
                    case TextBlock():
                        raise TypeError("OpenAI Responses cannot pass back signed text blocks")
                    case ToolCall(
                        tool_call_id=call_id,
                        name=name,
                        arguments=arguments,
                        signature=None,
                        has_provider_tool_call_id=True,
                    ):
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": arguments,
                            }
                        )
                    case ToolCall():
                        raise TypeError("OpenAI Responses cannot pass back provider-attached or synthetic tool calls")
                    case ReasoningRefBlock() if not allow_reference_reasoning:
                        raise NotImplementedError(
                            "Stateless OpenAI Responses passback requires encrypted reasoning content"
                        )
                    case ReasoningRefBlock(id=reasoning_id, content=content):
                        # Passed back by reference: the id is the handle to the stored
                        # reasoning item.
                        reasoning_item: dict[str, Any] = {"type": "reasoning", "id": reasoning_id, "summary": []}
                        if content:
                            reasoning_item["summary"] = [{"type": "summary_text", "text": content}]
                        input_items.append(reasoning_item)
                    case EncryptedReasoningBlock(
                        id=reasoning_id,
                        summary=summary,
                        encrypted_content=encrypted_content,
                    ):
                        # Stateless passback: the whole item — id, summary parts, and
                        # encrypted payload — is re-emitted verbatim in position.
                        input_items.append(
                            {
                                "type": "reasoning",
                                "id": reasoning_id,
                                "summary": [{"type": "summary_text", "text": part} for part in summary],
                                "encrypted_content": encrypted_content,
                            }
                        )
                    case ReasoningBlock():
                        raise NotImplementedError(
                            "OpenAI Responses cannot pass back reasoning without an item ID or encrypted content"
                        )
                    case (
                        SignedReasoningBlock()
                        | RedactedReasoningBlock()
                        | OpaqueBlock()
                        | ImageContentBlock()
                        | VideoContentBlock()
                        | AudioContentBlock()
                    ):
                        raise NotImplementedError(
                            f"OpenAI Responses cannot pass back {type(block).__name__} assistant blocks"
                        )
                    case _:
                        assert_never(block)
        elif isinstance(m, ToolMessage):
            # Tool results are function_call_output items
            if not isinstance(m.content, str):
                raise NotImplementedError(
                    "OpenAI Responses tool outputs currently require string content; multimodal tool output "
                    "needs a lossless provider mapping"
                )
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": m.tool_call_id,
                    "output": m.content,
                }
            )
        else:
            raise NotImplementedError(f"Unsupported message type: {type(m)}")

    return instructions, input_items


def _to_open_responses_request(
    msgs: Sequence[ChatMessage],
    *,
    use_provider_response_id: bool,
) -> tuple[str | None, str | None, list[dict[str, Any]]]:
    """Build one Responses request while preserving response-level continuation state.

    ``previous_response_id`` replaces only the earlier input items. System instructions
    are extracted from the complete local history because the Responses API does not
    carry a previous response's ``instructions`` into the next request.
    """
    instructions, _ = _to_open_responses_input([msg for msg in msgs if isinstance(msg, SystemMessage)])
    previous_response_id: str | None = None
    messages_to_process = msgs

    if use_provider_response_id:
        for index in range(len(msgs) - 1, -1, -1):
            msg = msgs[index]
            if isinstance(msg, AssistantMessage) and msg.provider_response_id is not None:
                if any(isinstance(block, ToolCall) and not block.has_provider_tool_call_id for block in msg.blocks):
                    raise NotImplementedError(
                        "Stored OpenAI Responses continuation cannot correlate a provider-idless tool call"
                    )
                previous_response_id = msg.provider_response_id
                messages_to_process = msgs[index + 1 :]
                break

    _, input_items = _to_open_responses_input(
        messages_to_process,
        allow_reference_reasoning=previous_response_id is not None,
    )
    return instructions, previous_response_id, input_items


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:  # noqa: ANN401
    """Get attribute from object or dict, with fallback default."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _parse_response_output(
    output: list[Any],
    *,
    allow_reference_reasoning: bool = True,
) -> list[AssistantBlock]:
    """Parse response output items into ordered assistant blocks.

    One exhaustive pass in item order: each ``message`` item becomes its own
    TextBlock (refusal content surfaces as answer text), each ``function_call``
    a ToolCall block, each ``reasoning`` item one of EncryptedReasoningBlock /
    ReasoningRefBlock / ReasoningBlock depending on the passback state it carries
    (an id is retained even when the summary is empty — it is the passback
    handle). Unknown item and content types raise until their semantics and
    passback behavior are explicitly implemented.
    """
    blocks: list[AssistantBlock] = []

    for item in output:
        item_type = _get_attr(item, "type")

        match item_type:
            case "message":
                text_parts: list[str] = []
                for content_item in _get_attr(item, "content", []):
                    content_type = _get_attr(content_item, "type")
                    if content_type == "refusal":
                        # A refusal is a normal API response, not malformed output: surface
                        # its text as the assistant's answer rather than crashing the turn.
                        refusal = _get_attr(content_item, "refusal")
                        if not isinstance(refusal, str):
                            raise ValueError(f"OpenAI Responses refusal is not a string: {refusal!r}")
                        text_parts.append(refusal)
                        continue
                    if content_type != "output_text":
                        raise NotImplementedError(
                            f"Unsupported OpenAI Responses message content type: {content_type!r}"
                        )
                    text = _get_attr(content_item, "text")
                    if not isinstance(text, str):
                        raise ValueError(f"OpenAI Responses output_text is not a string: {text!r}")
                    text_parts.append(text)
                if text_parts:
                    blocks.append(TextBlock(text="".join(text_parts)))

            case "function_call":
                blocks.append(
                    ToolCall.from_provider(
                        provider_id=_get_attr(item, "call_id"),
                        name=_get_attr(item, "name"),
                        arguments=_get_attr(item, "arguments", ""),
                    )
                )

            case "reasoning":
                # summary can be a list of Summary objects with .text attribute
                summary = _get_attr(item, "summary")
                if isinstance(summary, list):
                    summary_parts = []
                    for s in summary:
                        text = _get_attr(s, "text")
                        if not isinstance(text, str):
                            raise ValueError(f"OpenAI Responses reasoning summary entry has no text: {s!r}")
                        # Empty parts carry no data; dropping them keeps re-emission faithful.
                        if text:
                            summary_parts.append(text)
                elif summary:
                    summary_parts = [str(summary)]
                else:
                    thinking = _get_attr(item, "thinking")
                    summary_parts = [thinking] if thinking else []

                item_id = _get_attr(item, "id")
                encrypted_content = _get_attr(item, "encrypted_content")
                if encrypted_content and not item_id:
                    # The id is the passback handle; without it the encrypted payload
                    # cannot be re-emitted and ZDR reasoning state would silently vanish.
                    raise ValueError(f"OpenAI Responses reasoning item has encrypted_content but no id: {item!r}")
                if item_id and encrypted_content:
                    # Stateless (store=false / ZDR) item: carried whole for verbatim re-emission.
                    blocks.append(
                        EncryptedReasoningBlock(
                            id=item_id,
                            encrypted_content=encrypted_content,
                            summary=summary_parts,
                        )
                    )
                elif item_id:
                    if not allow_reference_reasoning:
                        raise NotImplementedError(
                            "A stateless OpenAI Responses call returned reference-only reasoning; "
                            "enable encrypted_reasoning so it can be passed back"
                        )
                    blocks.append(ReasoningRefBlock(id=item_id, content="\n".join(summary_parts)))
                elif summary_parts:
                    blocks.append(ReasoningBlock(content="\n".join(summary_parts)))

            case _:
                raise NotImplementedError(f"Unsupported OpenAI Responses output item type: {item_type!r}")

    return blocks


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
        encrypted_reasoning: bool = False,
        timeout: float | None = None,
        max_retries: int = 2,
        instructions: str | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize OpenAI SDK client with model configuration for Responses API.

        Args:
            model: Model identifier (e.g., 'gpt-4o', 'o1-preview').
            max_tokens: Maximum output tokens. Defaults to 64,000.
            base_url: API base URL. If None, uses OpenAI's standard URL.
                Use for OpenAI-compatible providers.
            api_key: API key for authentication. If None, reads from OPENROUTER_API_KEY
                environment variable.
            reasoning_effort: Reasoning effort level for extended thinking models
                (e.g., 'low', 'medium', 'high'). Only used with o1/o3 style models.
            encrypted_reasoning: Run stateless (``store=false``) and request
                ``include: ["reasoning.encrypted_content"]`` so reasoning items are
                captured as EncryptedReasoningBlock and re-emitted verbatim in
                position on passback. Required for zero-data-retention setups where
                the provider holds no conversation state.
            timeout: Request timeout in seconds. If None, uses OpenAI SDK default.
            max_retries: Number of retries for transient errors. Defaults to 2.
            instructions: Default system-level instructions. Can be overridden by
                SystemMessage in the messages list.
            kwargs: Additional arguments passed to responses.create(). Structural
                request keys owned by this client are rejected. ``tools`` /
                ``tool_choice`` and ``reasoning`` remain available for provider-native
                configuration, but cannot be combined with the corresponding dedicated
                ``generate(tools=...)`` or ``reasoning_effort`` configuration.
        """
        self._model = model
        self._max_tokens = max_tokens
        self._reasoning_effort = reasoning_effort
        self._encrypted_reasoning = encrypted_reasoning
        self._default_instructions = instructions
        self._kwargs = dict(kwargs or {})
        reserved_keys = sorted(_OWNED_REQUEST_KEYS & self._kwargs.keys())
        if reserved_keys:
            raise ValueError(
                f"OpenResponsesClient owns request keys {reserved_keys}; use its dedicated arguments and message "
                "history instead of kwargs"
            )

        # Initialize AsyncOpenAI client
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # Strip /responses suffix if present - SDK appends it automatically
        resolved_base_url = base_url
        if resolved_base_url and resolved_base_url.rstrip("/").endswith("/responses"):
            resolved_base_url = resolved_base_url.rstrip("/").removesuffix("/responses")

        self._client = AsyncOpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
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
    async def _create_response(self, request_kwargs: dict[str, Any]) -> Response:
        """Create one logical request with the client's standard transient retries."""
        return await self._client.responses.create(**request_kwargs)

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
        conditional_keys: set[str] = set()
        if tools:
            conditional_keys.update(("tools", "tool_choice"))
        if self._reasoning_effort:
            conditional_keys.add("reasoning")
        if conflicts := sorted(conditional_keys & self._kwargs.keys()):
            raise ValueError(f"OpenResponsesClient kwargs keys {conflicts} conflict with dedicated call configuration")

        # ``encrypted_reasoning`` owns statefulness; kwargs cannot override ``store``.
        stateful = not self._encrypted_reasoning

        # Convert messages to OpenResponses format. Stateful calls continue from the
        # latest provider response; stateless/ZDR calls replay encrypted reasoning items.
        instructions, previous_response_id, input_items = _to_open_responses_request(
            messages,
            use_provider_response_id=stateful,
        )

        # Use provided instructions or fall back to default
        final_instructions = instructions or self._default_instructions

        # Build request kwargs
        request_kwargs: dict[str, Any] = {
            **self._kwargs,
            "model": self._model,
            "input": input_items,
            "max_output_tokens": self._max_tokens,
        }
        if previous_response_id is not None:
            request_kwargs["previous_response_id"] = previous_response_id

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

        # Stateless mode: nothing is stored provider-side, so ask for the encrypted
        # reasoning payload and carry it client-side across turns. A caller-supplied
        # include list (via kwargs) is extended, not clobbered.
        if self._encrypted_reasoning:
            request_kwargs["store"] = False
            include = list(request_kwargs.get("include") or [])
            if "reasoning.encrypted_content" not in include:
                include.append("reasoning.encrypted_content")
            request_kwargs["include"] = include

        # Make API call
        request_start_time = perf_counter()
        try:
            response = await self._create_response(request_kwargs)
        except APIStatusError as error:
            if previous_response_id is None or not _is_missing_previous_response(error):
                raise
            try:
                replay_instructions, _, replay_input = _to_open_responses_request(
                    messages,
                    use_provider_response_id=False,
                )
            except (NotImplementedError, TypeError) as replay_error:
                raise RuntimeError(
                    "The stored OpenAI response is unavailable and this history cannot be replayed exactly "
                    f"without its provider-side continuation state: {replay_error}"
                ) from error

            logger.warning(
                "Stored OpenAI response %s was not found; retrying once with a full local-history replay",
                previous_response_id,
            )
            request_kwargs["input"] = replay_input
            request_kwargs.pop("previous_response_id", None)
            final_replay_instructions = replay_instructions or self._default_instructions
            if final_replay_instructions:
                request_kwargs["instructions"] = final_replay_instructions
            else:
                request_kwargs.pop("instructions", None)
            response = await self._create_response(request_kwargs)
        request_end_time = perf_counter()

        # Check for incomplete response (context overflow)
        if response.status == "incomplete":
            stop_reason = getattr(response, "incomplete_details", None)
            raise ContextOverflowError(
                f"Response incomplete for model {self.model_slug}: {stop_reason}. "
                "Reduce max_tokens or message length and try again."
            )
        if response.status != "completed":
            raise RuntimeError(
                f"OpenAI Responses returned unsupported status {response.status!r}: "
                f"{getattr(response, 'error', None)!r}"
            )

        provider_response_id: str | None = None
        if stateful:
            response_id = response.id
            if not isinstance(response_id, str) or not response_id:
                raise ValueError("Stored OpenAI Responses calls require a non-empty response ID")
            provider_response_id = response_id

        # Parse response output into ordered blocks
        blocks = _parse_response_output(response.output, allow_reference_reasoning=stateful)

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
            provider_response_id=provider_response_id,
            blocks=blocks,
            token_usage=TokenUsage(
                input=input_tokens,
                answer=answer_tokens,
                reasoning=reasoning_tokens,
            ),
            request_start_time=request_start_time,
            request_end_time=request_end_time,
        )
