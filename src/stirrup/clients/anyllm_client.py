"""any-llm-based LLM client for multi-provider support.

This client uses any-llm to provide a unified interface to multiple LLM providers
(OpenAI, Anthropic, Google, etc.) with automatic retries for transient failures.

Requires the anyllm extra: `pip install stirrup[anyllm]`
"""

import json
import logging
import warnings
from time import perf_counter
from typing import Any, Literal, cast

try:
    from anthropic.types import Message
    from any_llm import AnyLLM
    from any_llm.exceptions import ContextLengthExceededError, ProviderError, RateLimitError
    from any_llm.providers.anthropic.anthropic import AnthropicProvider
    from any_llm.types.completion import ChatCompletion
    from any_llm.types.completion import ReasoningEffort as AnyLLMReasoningEffort
    from any_llm.utils.exception_handler import convert_exception
except ImportError as e:
    raise ImportError(
        "Requires installation of the anyllm extra. "
        "Install with: `uv pip install stirrup[anyllm]` or `uv add stirrup[anyllm]`"
    ) from e

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from stirrup.clients.utils import to_openai_messages, to_openai_tools, validate_token_budgets
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError
from stirrup.core.models import (
    AssistantBlock,
    AssistantMessage,
    ChatMessage,
    LLMClient,
    ReasoningBlock,
    RedactedReasoningBlock,
    SignedReasoningBlock,
    TextBlock,
    TokenUsage,
    Tool,
    ToolCall,
)

__all__ = [
    "AnyLLMClient",
]

LOGGER = logging.getLogger(__name__)

type ReasoningEffort = AnyLLMReasoningEffort | Literal["default"]


class _AnthropicProvider(AnthropicProvider):
    """Preserve individual signed/redacted thinking blocks across tool turns."""

    @staticmethod
    def _convert_completion_response(response: Message) -> ChatCompletion:
        completion = AnthropicProvider._convert_completion_response(response)  # noqa: SLF001
        message = completion.choices[0].message
        if message.extra_content is None:
            message.extra_content = {}
        message.extra_content.setdefault("anthropic", {})["thinking_blocks"] = [
            block.model_dump() for block in response.content if block.type in ("thinking", "redacted_thinking")
        ]
        return completion


def _to_anyllm_messages(messages: list[ChatMessage], provider: str) -> list[dict[str, Any]]:
    """Translate the shared chat format's reasoning metadata to any-llm."""
    wire = to_openai_messages(messages, allow_tool_call_signatures=True)
    for message in wire:
        if provider == "gemini" and message["role"] == "system" and isinstance(message["content"], list):
            message["content"] = "".join(part["text"] for part in message["content"])
        for tool_call in message.get("tool_calls", []):
            metadata = tool_call.pop("provider_specific_fields", {})
            if signature := metadata.get("thought_signature"):
                tool_call["extra_content"] = {"google": {"thought_signature": signature}}
        thinking_blocks = message.pop("thinking_blocks", [])
        if provider == "anthropic" and message["role"] == "assistant":
            # Pass native blocks through any-llm without merging thinking signatures.
            content = [*thinking_blocks, *(message.get("content") or [])]
            content.extend(
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"] or "{}"),
                }
                for tool_call in message.pop("tool_calls", [])
            )
            message["content"] = content
            message.pop("reasoning_content", None)
    return wire


def _parse_thinking_blocks(
    thinking_blocks: list[dict[str, Any]] | None,
) -> list[SignedReasoningBlock | RedactedReasoningBlock | ReasoningBlock]:
    """Parse native thinking blocks into reasoning blocks, one per entry, in order."""
    parsed: list[SignedReasoningBlock | RedactedReasoningBlock | ReasoningBlock] = []
    for entry in thinking_blocks or []:
        if entry.get("type") == "redacted_thinking":
            parsed.append(RedactedReasoningBlock(data=entry["data"]))
            continue
        signature = entry.get("signature") or entry.get("thinking_signature")
        content = entry.get("thinking")
        if signature is not None:
            parsed.append(SignedReasoningBlock(signature=signature, content=content or ""))
        elif content is not None:
            parsed.append(ReasoningBlock(content=content))
        else:
            raise ValueError(f"Signature and content not found in the thinking block response: {entry!r}")
    return parsed


class AnyLLMClient(LLMClient):
    """any-llm-based client supporting multiple LLM providers with unified interface.

    Includes automatic retries for transient failures and token usage tracking.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 64_000,
        *,
        context_window_tokens: int,
        model_slug: str | None = None,
        api_key: str | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize any-llm client with model configuration and capabilities.

        Args:
            model: Model identifier for any-llm (e.g., 'openai:gpt-5.5'). Legacy
                'provider/model' names are also accepted; bare names use OpenAI.
            max_tokens: Maximum number of tokens the provider may generate
            context_window_tokens: Context capacity used to decide when Agent history
                should be summarized.
            model_slug: Deprecated. Use model instead.
            reasoning_effort: Reasoning effort level for extended thinking models (e.g., 'medium', 'high')
            kwargs: Additional arguments to pass to any-llm completion calls.
                Provider SDK options can be supplied in ``client_args``.
                Keys that collide with the arguments this client sets (including
                ``max_tokens``) raise TypeError at request time; other keys pass
                through unvalidated.

        Raises:
            ValueError: If no model is provided, ``context_window_tokens`` is not
                positive, or ``max_tokens`` exceeds it.
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
        validate_token_budgets(max_tokens, context_window_tokens)

        self._model = model
        self._max_tokens = max_tokens
        self._context_window_tokens = context_window_tokens
        self._reasoning_effort = "auto" if reasoning_effort == "default" else reasoning_effort
        self._api_key = api_key
        self._provider, separator, self._model_id = model.partition(":" if ":" in model.split("/", 1)[0] else "/")
        if not separator:
            self._provider, self._model_id = "openai", model
        self._kwargs = (kwargs or {}).copy()
        self._client_args: dict[str, Any] = self._kwargs.pop("client_args", {})
        if self._provider in ("openai", "anthropic", "openrouter"):
            self._client_args.setdefault("max_retries", 0)
        self._api_base = self._kwargs.pop("api_base", None)
        if self._provider == "anthropic":
            self._kwargs.setdefault("timeout", 600)
        self._client: AnyLLM | None = None

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
        """Model identifier used by any-llm."""
        return self._model

    @retry(
        retry=retry_if_exception_type((ProviderError, RateLimitError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        """Generate assistant response with optional tool calls. Retries up to 3 times on timeout/connection errors."""
        request_start_time = perf_counter()
        if self._client is None:
            if self._provider == "anthropic":
                self._client = _AnthropicProvider(api_key=self._api_key, api_base=self._api_base, **self._client_args)
            else:
                self._client = AnyLLM.create(
                    self._provider, api_key=self._api_key, api_base=self._api_base, **self._client_args
                )
        try:
            r = await self._client.acompletion(
                model=self._model_id,
                messages=cast(Any, _to_anyllm_messages(messages, self._provider)),
                tools=to_openai_tools(tools) if tools else None,
                tool_choice="auto" if tools else None,
                max_tokens=self._max_tokens,
                reasoning_effort=self._reasoning_effort,
                stream=False,
                **self._kwargs,
            )
        except (TypeError, ValueError):
            raise
        except Exception as e:
            error = convert_exception(e, self._provider)
            if isinstance(error, ContextLengthExceededError):
                raise ContextOverflowError(str(error)) from e
            if error is e:
                raise
            raise error from e
        request_end_time = perf_counter()

        choice = r.choices[0]

        if choice.finish_reason in ("max_tokens", "length"):
            raise OutputTokenLimitError(
                model_slug=self.model_slug,
                max_tokens=self._max_tokens,
                provider_reason=choice.finish_reason,
            )

        msg = choice.message

        # Chat Completions is channel-shaped on the wire — no ordering to preserve —
        # so blocks are built in canonical channel order: reasoning → text → tool calls.
        extra = msg.extra_content or {}
        blocks: list[AssistantBlock] = list(_parse_thinking_blocks(extra.get("anthropic", {}).get("thinking_blocks")))
        if not blocks and msg.reasoning is not None:
            blocks.append(ReasoningBlock(content=msg.reasoning.content))

        content = msg.content or ""
        if content:
            blocks.append(TextBlock(text=content))

        blocks.extend(
            ToolCall.from_provider(
                provider_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "",
                signature=(getattr(tc, "extra_content", None) or {}).get("google", {}).get("thought_signature"),
            )
            for tc in (msg.tool_calls or [])
            if tc.type == "function"
        )

        usage = r.usage
        if usage is None:
            raise ValueError("Provider response did not include token usage")
        input_tokens = usage.prompt_tokens
        reasoning_tokens = 0
        if usage.completion_tokens_details:
            reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0
        output_tokens = usage.completion_tokens
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
