"""Custom exceptions for agent framework."""

__all__ = ["ContextOverflowError", "IncompleteResponseError", "OutputTokenLimitError"]


class ContextOverflowError(Exception):
    """Raised when request input exceeds the model's context capacity."""


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
