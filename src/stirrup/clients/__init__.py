"""LLM client implementations.

The default client is ChatCompletionsClient, which uses the OpenAI SDK directly
and supports any OpenAI-compatible API via the `base_url` parameter.

OpenResponsesClient uses the OpenAI Responses API (responses.create) for providers
that support this newer API format.

For multi-provider support via LiteLLM, install the litellm extra:
    pip install stirrup[litellm]
"""

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.clients.open_responses_client import OpenResponsesClient

__all__ = [
    "ChatCompletionsClient",
    "OpenResponsesClient",
]
