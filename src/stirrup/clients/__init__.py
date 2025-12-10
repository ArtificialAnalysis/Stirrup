"""LLM client implementations.

The default client is ChatCompletionsClient, which uses the OpenAI SDK directly
and supports any OpenAI-compatible API via the `base_url` parameter.

For multi-provider support via LiteLLM, install the litellm extra:
    pip install stirrup[litellm]
"""

from stirrup.clients.chat_completions_client import ChatCompletionsClient

__all__ = [
    "ChatCompletionsClient",
]
