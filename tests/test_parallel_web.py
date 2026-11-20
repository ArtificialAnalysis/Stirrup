"""Opt-in Parallel search through the normal agent loop and MCP HTTP transport."""

import json
from collections.abc import Awaitable
from typing import Any, cast
from unittest.mock import AsyncMock

import anyio
import httpx
import pytest
from pydantic import ValidationError
from pytest import MonkeyPatch

pytest.importorskip("mcp")

from stirrup import Agent
from stirrup.core.models import AssistantMessage, ChatMessage, Tool, ToolCall, ToolMessage, ToolResult
from stirrup.tools import WebToolProvider, default_tools, parallel_web
from stirrup.tools.parallel_web import ApprovedSearchParams
from tests.test_agent import MockLLMClient

PUBLIC_QUERY = "Python 3.13 tomllib TOMLDecodeError"
SOURCE_URL = "https://docs.python.org/3.13/library/tomllib.html"
EVIDENCE = {"results": [{"url": SOURCE_URL, "excerpts": ["Invalid TOML raises TOMLDecodeError."]}]}


class MCPFixture:
    """Use the supported httpx transport seam; keep the real MCP SDK and protocol."""

    def __init__(self, monkeypatch: MonkeyPatch, result: dict[str, Any] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.calls: list[dict[str, Any]] = []
        self.clients: list[httpx.AsyncClient] = []
        self.result = result if result is not None else {"content": [{"type": "text", "text": json.dumps(EVIDENCE)}]}
        self.delay = False
        self.unavailable = False
        real_client = httpx.AsyncClient

        def factory(**kwargs: object) -> httpx.AsyncClient:
            client = real_client(transport=httpx.MockTransport(self.handle), **cast(Any, kwargs))
            self.clients.append(client)
            return client

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert str(request.url) == parallel_web.PARALLEL_MCP_URL
        assert request.headers["User-Agent"] == parallel_web.PARALLEL_USER_AGENT
        assert "authorization" not in request.headers
        assert "x-api-key" not in request.headers
        if self.unavailable:
            return httpx.Response(503)
        data = json.loads(request.content)
        method = data["method"]
        if method == "initialize":
            result = {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "notifications/initialized":
            return httpx.Response(202)
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            self.calls.append(data["params"])
            if self.delay:
                await anyio.sleep_forever()
            result = self.result
        else:
            raise AssertionError(method)
        return httpx.Response(
            200,
            headers={"Content-Type": "application/json"},
            json={"jsonrpc": "2.0", "id": data["id"], "result": result},
        )


async def execute(tool: Tool) -> ToolResult:
    return await cast(Awaitable[ToolResult], tool.executor(ApprovedSearchParams()))


@pytest.mark.parametrize("query", [None, "", "   "])
def test_parallel_requires_approved_query(query: str | None) -> None:
    with pytest.raises(ValueError, match="user-approved"):
        WebToolProvider(search_provider="parallel", parallel_public_query=query)


def test_query_does_not_implicitly_select_parallel() -> None:
    with pytest.raises(ValueError, match="explicit"):
        WebToolProvider(parallel_public_query=PUBLIC_QUERY)
    with pytest.raises(ValueError, match="search_provider"):
        WebToolProvider(search_provider="unknown")  # ty: ignore[invalid-argument-type]


async def test_defaults_and_brave_never_use_parallel(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    parallel = AsyncMock(side_effect=AssertionError("Parallel was not selected"))
    monkeypatch.setattr(parallel_web, "streamablehttp_client", parallel)
    provider = next(p for p in default_tools() if isinstance(p, WebToolProvider))
    async with provider as tools:
        assert [tool.name for tool in tools] == ["fetch_web_page"]
    async with WebToolProvider(brave_api_key="fixture-key") as tools:
        search = next(t for t in tools if t.name == "web_search")
        assert "Brave" in search.description
        assert "query" in search.parameters.model_fields
    parallel.assert_not_called()


async def test_exact_query_one_attempt_across_tools_and_provider_reentry(monkeypatch: MonkeyPatch) -> None:
    fixture = MCPFixture(monkeypatch)
    # Preserve the approved literal, including whitespace; do not normalize it.
    literal = f" {PUBLIC_QUERY} "
    provider = WebToolProvider(search_provider="parallel", parallel_public_query=literal, brave_api_key="unused")
    async with provider as tools:
        tool = next(t for t in tools if t.name == "web_search")
        assert tool.parameters.model_json_schema()["additionalProperties"] is False
        assert tool.parameters.model_json_schema()["properties"] == {}
        assert PUBLIC_QUERY not in tool.description
        for replacement in (
            {"query": "private prompt"},
            {"search_queries": ["private prompt"]},
            {"objective": "other"},
        ):
            with pytest.raises(ValidationError):
                tool.parameters.model_validate(replacement)
        assert fixture.requests == []
        result = await execute(tool)
        assert result.success
        assert SOURCE_URL in result.content
        assert "TOMLDecodeError" in result.content
        second_tool = next(t for t in provider.get_tools() if t.name == "web_search")
        assert not (await execute(second_tool)).success
    async with provider as tools:
        assert not (await execute(next(t for t in tools if t.name == "web_search"))).success
    assert fixture.calls == [
        {
            "name": "web_search",
            "arguments": {
                "objective": "Find official public documentation for the approved keyword query.",
                "search_queries": [literal],
            },
        }
    ]
    assert all(client.is_closed for client in fixture.clients)


@pytest.mark.parametrize("case", ["empty", "empty_results", "structured", "tool_error", "unavailable", "timeout"])
async def test_result_and_failure_behavior(monkeypatch: MonkeyPatch, case: str) -> None:
    results = {
        "empty": {"content": []},
        "empty_results": {"content": [], "structuredContent": {"results": []}},
        "structured": {"content": [], "structuredContent": EVIDENCE},
        "tool_error": {"content": [{"type": "text", "text": "No search available"}], "isError": True},
    }
    fixture = MCPFixture(monkeypatch, results.get(case))
    fixture.delay = case == "timeout"
    fixture.unavailable = case == "unavailable"
    async with WebToolProvider(search_provider="parallel", parallel_public_query=PUBLIC_QUERY, timeout=0.2) as tools:
        tool = next(t for t in tools if t.name == "web_search")
        result = await execute(tool)
        assert result.success == (case in {"empty", "empty_results", "structured"})
        if case == "empty":
            assert "no search results" in result.content
        elif case == "empty_results":
            assert json.loads(str(result.content))["results"] == []
        elif case == "structured":
            assert SOURCE_URL in result.content
        elif case == "timeout":
            assert "timed out" in result.content
        elif case == "unavailable":
            assert "unavailable" in result.content
        assert not (await execute(tool)).success
    assert len(fixture.calls) <= 1
    assert all(client.is_closed for client in fixture.clients)


async def test_agent_invokes_scoped_search_and_cites_received_fact(monkeypatch: MonkeyPatch) -> None:
    fixture = MCPFixture(monkeypatch)

    class EvidenceClient(MockLLMClient):
        async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
            assert tools["web_search"].parameters.model_json_schema()["properties"] == {}
            outputs = [m for m in messages if isinstance(m, ToolMessage) and m.name == "web_search"]
            if not outputs:
                # Try a model-authored replacement first: Agent must reject it before dispatch.
                arguments = '{"query":"private input must stay local"}'
            elif not outputs[-1].args_was_valid:
                assert fixture.calls == []
                arguments = "{}"
            else:
                assert outputs[-1].success
                evidence = json.loads(str(outputs[-1].content))["results"][0]
                fact = evidence["excerpts"][0]
                answer = f"{fact} [Python documentation]({evidence['url']})"
                return AssistantMessage(
                    blocks=[
                        ToolCall(
                            name="finish", arguments=json.dumps({"reason": answer, "paths": []}), tool_call_id="finish"
                        )
                    ]
                )
            return AssistantMessage(
                blocks=[ToolCall(name="web_search", arguments=arguments, tool_call_id=str(len(outputs)))]
            )

    agent = Agent(
        client=EvidenceClient([]),
        name="attended-docs",
        max_turns=4,
        tools=[WebToolProvider(search_provider="parallel", parallel_public_query=PUBLIC_QUERY)],
    )
    async with agent.session(cache_on_interrupt=False) as session:
        finish, history, _ = await session.run(
            "Use the approved lookup to explain invalid TOML handling and cite its source."
        )
    assert finish is not None
    assert "TOMLDecodeError" in finish.reason
    assert f"]({SOURCE_URL})" in finish.reason
    assert len(fixture.calls) == 1
    assert any(isinstance(m, ToolMessage) and SOURCE_URL in m.content for turn in history for m in turn)


async def test_cancelled_attempt_cannot_be_reissued(monkeypatch: MonkeyPatch) -> None:
    fixture = MCPFixture(monkeypatch)
    fixture.delay = True
    async with WebToolProvider(search_provider="parallel", parallel_public_query=PUBLIC_QUERY) as tools:
        tool = next(t for t in tools if t.name == "web_search")
        with anyio.move_on_after(0.1) as scope:
            await execute(tool)
        assert scope.cancel_called
        assert not (await execute(tool)).success
    assert len(fixture.calls) == 1
    assert all(client.is_closed for client in fixture.clients)
