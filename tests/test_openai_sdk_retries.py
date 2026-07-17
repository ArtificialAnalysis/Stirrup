"""Behavioral tests for OpenAI SDK retry configuration."""

import httpx
import pytest
from openai import InternalServerError

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.clients.open_responses_client import OpenResponsesClient
from stirrup.core.models import UserMessage


@pytest.mark.parametrize(
    "client_class",
    [
        pytest.param(ChatCompletionsClient, id="chat_completions"),
        pytest.param(OpenResponsesClient, id="open_responses"),
    ],
)
async def test_max_retries_controls_sdk_request_count(
    client_class: type[ChatCompletionsClient] | type[OpenResponsesClient],
) -> None:
    request_count = 0

    def fail(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            500,
            headers={"retry-after-ms": "1"},
            json={"error": {"message": "retry", "type": "server_error"}},
            request=request,
        )

    client = client_class(model="gpt-4o", api_key="test-key", max_retries=1)
    original_sdk_client = client._client  # noqa: SLF001
    client._client = original_sdk_client.with_options(  # noqa: SLF001
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fail))
    )
    await original_sdk_client.close()

    try:
        with pytest.raises(InternalServerError):
            await client.generate([UserMessage(content="hello")], {})
    finally:
        await client._client.close()  # noqa: SLF001

    assert request_count == 2
