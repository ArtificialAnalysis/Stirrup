"""Behavioural contracts for Agent session teardown.

Teardown owns real resources - containers, cloud sandboxes, MCP subprocesses, host temp
directories - so it must complete under cancellation and must not let one failing phase
skip the next or hide the failure the caller is diagnosing.
"""

import shutil
import tempfile
from pathlib import Path

import anyio
import pytest
from anyio.lowlevel import checkpoint
from pydantic import BaseModel

from stirrup.core.agent import Agent
from stirrup.core.models import AssistantMessage, ChatMessage, LLMClient, Tool, ToolProvider, ToolResult
from stirrup.utils.logging import AgentLogger


class UnusedClient(LLMClient):
    """Client for sessions that never run a turn."""

    @property
    def model_slug(self) -> str:
        return "unused-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        del messages, tools
        raise AssertionError("This test never runs an agent turn")


def _noop_tool(name: str) -> Tool:
    def noop(_params: BaseModel) -> ToolResult:
        return ToolResult(content="ok")

    return Tool(name=name, description="Does nothing", executor=noop)


class TempDirProvider(ToolProvider):
    """Provider owning a host temp directory, released only after an await.

    Stands in for the execution-environment providers whose teardown awaits: Docker stops a
    container, E2B kills a sandbox, MCP shuts a subprocess down. LocalCodeExecToolProvider
    cannot substitute here - its teardown never awaits, so cancellation cannot interrupt it.
    """

    def __init__(self) -> None:
        self.temp_dir: Path | None = None
        self.exit_count = 0

    async def __aenter__(self) -> Tool:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="stirrup-teardown-test-"))
        return _noop_tool("temp_dir_tool")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        del exc_type, exc_val, exc_tb
        await checkpoint()
        if self.temp_dir is not None:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
        self.exit_count += 1


class BlockingEntryProvider(ToolProvider):
    """Provider that blocks forever while starting up, so entry can be cancelled."""

    def __init__(self) -> None:
        self.entry_started = anyio.Event()

    async def __aenter__(self) -> Tool:
        self.entry_started.set()
        await anyio.sleep_forever()
        raise AssertionError("unreachable")


class FailingEntryProvider(ToolProvider):
    def __init__(self) -> None:
        self.entry_error = RuntimeError("provider startup failed")

    async def __aenter__(self) -> Tool:
        raise self.entry_error


class CleanupFailingProvider(ToolProvider):
    def __init__(self) -> None:
        self.exit_count = 0
        self.exit_error = RuntimeError("provider cleanup failed")

    async def __aenter__(self) -> Tool:
        return _noop_tool("cleanup_failing_tool")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.exit_count += 1
        raise self.exit_error


class CleanupFailingLogger(AgentLogger):
    def __init__(self) -> None:
        super().__init__(show_spinner=False)
        self.exit_error = RuntimeError("logger cleanup failed")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        raise self.exit_error


def _quiet_logger() -> AgentLogger:
    return AgentLogger(show_spinner=False)


def _cleanup_notes(exception: BaseException) -> list[str]:
    return list(getattr(exception, "__notes__", []))


async def test_cancelled_session_still_releases_providers() -> None:
    environment = TempDirProvider()
    agent = Agent(client=UnusedClient(), name="cancel-exit", tools=[environment], logger=_quiet_logger())

    with anyio.CancelScope() as scope:
        async with agent.session(cache_on_interrupt=False):
            scope.cancel()
            await anyio.sleep_forever()

    assert scope.cancelled_caught
    assert environment.exit_count == 1
    assert environment.temp_dir is None


async def test_cancelled_session_entry_still_releases_entered_providers() -> None:
    environment = TempDirProvider()
    blocking = BlockingEntryProvider()
    agent = Agent(
        client=UnusedClient(),
        name="cancel-entry",
        tools=[environment, blocking],
        logger=_quiet_logger(),
    )

    async def enter_session() -> None:
        async with agent.session(cache_on_interrupt=False):
            raise AssertionError("entry should not complete")

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(enter_session)
        await blocking.entry_started.wait()
        task_group.cancel_scope.cancel()

    assert environment.exit_count == 1
    assert environment.temp_dir is None


async def test_body_failure_survives_logger_and_provider_cleanup_failures() -> None:
    environment = TempDirProvider()
    flaky = CleanupFailingProvider()
    agent = Agent(
        client=UnusedClient(),
        name="body-failure",
        tools=[environment, flaky],
        logger=CleanupFailingLogger(),
    )
    body_error = RuntimeError("the failure the operator is debugging")

    with pytest.raises(RuntimeError) as exc_info:
        async with agent.session(cache_on_interrupt=False):
            raise body_error

    assert exc_info.value is body_error
    assert environment.exit_count == 1
    assert environment.temp_dir is None
    assert flaky.exit_count == 1
    notes = _cleanup_notes(body_error)
    assert any("Logger failed during session cleanup" in note for note in notes)
    assert any("Tool provider failed during session cleanup" in note for note in notes)


async def test_cleanup_failure_surfaces_when_the_session_body_succeeded() -> None:
    flaky = CleanupFailingProvider()
    failing_logger = CleanupFailingLogger()
    agent = Agent(client=UnusedClient(), name="cleanup-only", tools=[flaky], logger=failing_logger)

    with pytest.raises(RuntimeError) as exc_info:
        async with agent.session(cache_on_interrupt=False):
            pass

    assert exc_info.value is failing_logger.exit_error
    assert flaky.exit_count == 1
    assert any("Tool provider failed during session cleanup" in note for note in _cleanup_notes(exc_info.value))


async def test_entry_failure_is_not_masked_by_a_cleanup_failure() -> None:
    flaky = CleanupFailingProvider()
    failing_entry = FailingEntryProvider()
    agent = Agent(
        client=UnusedClient(),
        name="entry-failure",
        tools=[flaky, failing_entry],
        logger=_quiet_logger(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        async with agent.session(cache_on_interrupt=False):
            pass

    assert exc_info.value is failing_entry.entry_error
    assert flaky.exit_count == 1
    assert any("Tool provider failed during session cleanup" in note for note in _cleanup_notes(exc_info.value))
