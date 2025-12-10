"""Custom exceptions for agent framework."""

__all__ = ["ContextOverflowError"]


class ContextOverflowError(Exception):
    """Raised when LLM context window is exceeded (max_tokens or length finish_reason)."""
