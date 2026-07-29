"""Custom exceptions for agent framework."""

__all__ = ["CacheUnusableError", "ContextOverflowError"]


class ContextOverflowError(Exception):
    """Raised when LLM context window is exceeded (max_tokens or length finish_reason)."""


class CacheUnusableError(Exception):
    """Raised when a cache exists for a task identity but cannot be trusted to resume from."""
