"""Custom exceptions for agent framework."""

from typing import Literal

__all__ = [
    "CompactionReason",
    "CompactionRequiredError",
    "ContextOverflowError",
    "IncompleteResponseError",
    "OutputTokenLimitError",
]

type CompactionReason = Literal["image_count", "request_size"]


class CompactionRequiredError(Exception):
    """Raised by an ``LLMClient`` when compacting the conversation could fix a rejected request.

    For limits other than tokens: too many images (``"image_count"``) or too many request
    bytes (``"request_size"``). The agent drops the latest turn, summarizes the rest and
    retries the step once, as for ``ContextOverflowError``. ``reason`` is recorded as the
    summary's ``trigger``.
    """

    def __init__(self, *args: object, reason: CompactionReason) -> None:
        super().__init__(*args)
        self.reason = reason


class ContextOverflowError(Exception):
    """Raised when request input exceeds the model's context capacity.

    The agent drops the latest turn, summarizes the rest and retries the step once.
    """


class IncompleteResponseError(Exception):
    """Raised when a provider returns an incomplete response Stirrup cannot recover from.

    Covers stop reasons other than the context and output-budget cases, such as a
    content filter. Retrying the same request is not expected to help.
    """


class OutputTokenLimitError(Exception):
    """Raised when a provider exhausts the configured response-token budget."""

    def __init__(self, *, model_slug: str, max_tokens: int, provider_reason: str) -> None:
        self.model_slug = model_slug
        self.max_tokens = max_tokens
        self.provider_reason = provider_reason
        super().__init__(
            f"Model '{model_slug}' exhausted its configured output budget "
            f"(max_tokens={max_tokens}; provider reason: {provider_reason}). "
            "Increase max_tokens or ask the model for a shorter response."
        )
