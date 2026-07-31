"""Tests for agent core functionality."""

import json
from collections.abc import Awaitable, Callable
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image
from pydantic import BaseModel

from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core.agent import Agent, SubAgentParams
from stirrup.core.cache import CacheState
from stirrup.core.exceptions import ContextOverflowError, OutputTokenLimitError
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    ImageContentBlock,
    LLMClient,
    SummaryMessage,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolResult,
    TurnWarningMessage,
    UserMessage,
)
from stirrup.tools import default_tools
from stirrup.tools.finish import SIMPLE_FINISH_TOOL, FinishParams


class MockLLMClient(LLMClient):
    """Mock LLM client for testing. context_window_tokens matches max_tokens unless given."""

    def __init__(
        self,
        responses: list[AssistantMessage | Exception],
        max_tokens: int = 100_000,
        context_window_tokens: int | None = None,
    ) -> None:
        self.responses = responses
        self.call_count = 0
        self._context_window_tokens = max_tokens if context_window_tokens is None else context_window_tokens
        self.tools_seen: list[dict[str, Tool]] = []

    @property
    def model_slug(self) -> str:
        return "mock-model"

    @property
    def context_window_tokens(self) -> int:
        return self._context_window_tokens

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:  # noqa: ARG002
        self.tools_seen.append(tools)
        response = self.responses[self.call_count]
        self.call_count += 1
        if isinstance(response, Exception):
            raise response
        return response


async def _finish_without_file_validation(params: FinishParams) -> ToolResult[None]:
    return ToolResult(content=params.reason)


UNVALIDATED_FINISH_TOOL = Tool[FinishParams, None](
    name=DEFAULT_FINISH_TOOL_NAME,
    description="Finish without validating declared paths.",
    parameters=FinishParams,
    executor=_finish_without_file_validation,
)


class RecordingParams(BaseModel):
    label: str


def recording_tool(name: str, executions: list[str], *, success: bool = True) -> Tool[RecordingParams, None]:
    def record(params: RecordingParams) -> ToolResult:
        executions.append(params.label)
        return ToolResult(content=params.label, success=success)

    return Tool[RecordingParams, None](
        name=name,
        description="Record an execution label",
        parameters=RecordingParams,
        executor=record,  # ty: ignore[invalid-argument-type]
    )


def _sample_png_block() -> ImageContentBlock:
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return ImageContentBlock(data=buffer.getvalue())


async def test_agent_basic_finish() -> None:
    """Test agent completes successfully when finish tool is called."""
    # Create mock responses
    responses = [
        AssistantMessage(
            content="I'll finish now",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Task completed successfully", "paths": []}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
            request_start_time=100.0,
            request_end_time=100.4,
        )
    ]

    # Create agent with mock client
    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    # Run agent with session context
    async with agent.session() as session:
        finish_params, message_history, run_metadata = await session.run(
            [
                SystemMessage(content="Test system message"),
                UserMessage(content="Test task"),
            ]
        )

    # Assertions
    assert finish_params is not None
    assert isinstance(finish_params, FinishParams)
    assert finish_params.reason == "Task completed successfully"
    assert isinstance(run_metadata, dict)
    # Agent's own token usage metadata should be present
    assert "token_usage" in run_metadata
    assert len(message_history) == 1  # One turn
    assert client.call_count == 1


async def test_output_limit_does_not_control_context_summarization() -> None:
    responses = [
        AssistantMessage(content="Working", tool_calls=[], token_usage=TokenUsage(input=700, answer=100)),
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "done", "paths": []}',
                    tool_call_id="finish",
                )
            ],
            token_usage=TokenUsage(),
        ),
    ]
    client = MockLLMClient(responses, max_tokens=100, context_window_tokens=64_000)
    agent = Agent(
        client=client,
        name="separate-budget-agent",
        max_turns=2,
        tools=[],
        context_summarization_cutoff=0.7,
    )

    async with agent.session(cache_on_interrupt=False) as session:
        finish_params, _history, _metadata = await session.run("complete the task")

    assert finish_params is not None
    assert client.call_count == 2


def test_agent_rejects_structural_client_without_context_window_tokens() -> None:
    class DuckTypedLegacyClient:
        """Structural client with no context_window_tokens attribute at all."""

        @property
        def model_slug(self) -> str:
            return "duck-model"

        @property
        def max_tokens(self) -> int:
            return 1_000

        async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
            del messages, tools
            raise NotImplementedError

    with pytest.raises(ValueError, match="context_window_tokens must be a positive int"):
        Agent(client=DuckTypedLegacyClient(), name="duck-client-agent", tools=[])  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("context_window_tokens", [0, True, 64_000.0])
def test_agent_rejects_context_window_that_is_not_a_positive_int(context_window_tokens: int | float) -> None:
    client = MockLLMClient([], max_tokens=1_000, context_window_tokens=context_window_tokens)  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="context_window_tokens must be a positive int"):
        Agent(client=client, name="invalid-context-agent", tools=[])


async def test_output_token_limit_surfaces_without_agent_retry_or_recovery() -> None:
    error = OutputTokenLimitError(model_slug="mock-model", max_tokens=100, provider_reason="length")
    client = MockLLMClient(
        [error, AssistantMessage(content="unused", tool_calls=[], token_usage=TokenUsage())],
        max_tokens=100,
    )
    agent = Agent(client=client, name="output-limit-agent", tools=[])

    async with agent.session(cache_on_interrupt=False) as session:
        with pytest.raises(OutputTokenLimitError) as exc_info:
            await session.run("complete the task")

    assert exc_info.value is error
    assert client.call_count == 1


async def test_finish_runs_last_however_the_provider_ordered_the_calls(tmp_path: Path) -> None:
    """The finish call ends the turn, so it executes after every ordinary call in that turn.

    The finish tool validates that its declared file exists, the way SIMPLE_FINISH_TOOL does: the
    provider put the finish first, so executing calls in array order would fail it on a file the
    same turn goes on to write.
    """
    executions: list[str] = []

    class PathParams(BaseModel):
        name: str

    def write_file(params: PathParams) -> ToolResult:
        (tmp_path / params.name).write_text("report")
        executions.append(params.name)
        return ToolResult(content=f"Wrote {params.name}")

    def finish(params: PathParams) -> ToolResult:
        executions.append("finish")
        if not (tmp_path / params.name).exists():
            return ToolResult(content=f"File does not exist: {params.name}", success=False)
        return ToolResult(content="Done")

    def path_tool(name: str, executor: Callable[[PathParams], ToolResult]) -> Tool[PathParams, None]:
        return Tool[PathParams, None](
            name=name,
            description=name,
            parameters=PathParams,
            executor=executor,  # ty: ignore[invalid-argument-type]
        )

    response = AssistantMessage(
        content="Finish first, but the other work must still happen",
        tool_calls=[
            ToolCall(name="finish", arguments='{"name": "report.md"}', tool_call_id="call_finish"),
            ToolCall(name="write_file", arguments='{"name": "report.md"}', tool_call_id="call_report"),
            ToolCall(name="write_file", arguments='{"name": "notes.md"}', tool_call_id="call_notes"),
        ],
        token_usage=TokenUsage(),
    )
    agent = Agent(
        client=MockLLMClient([response]),
        name="test-agent",
        tools=[path_tool("write_file", write_file)],
        finish_tool=path_tool("finish", finish),
        run_sync_in_thread=False,
    )

    _, tool_messages, finish_params = await agent.step([UserMessage(content="Test task")], {})

    assert executions == ["report.md", "notes.md", "finish"]
    assert finish_params == PathParams(name="report.md")
    # Every call is answered exactly once, so the history stays 1:1 with the assistant message.
    assert [message.tool_call_id for message in tool_messages] == ["call_report", "call_notes", "call_finish"]
    assert all(message.success for message in tool_messages)


@pytest.mark.parametrize(
    ("finish_arguments", "finish_succeeds", "expected_executions"),
    [
        pytest.param("{}", True, ["ordinary"], id="invalid-arguments"),
        pytest.param('{"label": "finish"}', False, ["ordinary", "finish"], id="unsuccessful"),
    ],
)
async def test_unsuccessful_finish_does_not_terminate_the_run(
    finish_arguments: str,
    finish_succeeds: bool,
    expected_executions: list[str],
) -> None:
    executions: list[str] = []
    record = recording_tool("record", executions)
    finish = recording_tool("finish", executions, success=finish_succeeds)
    response = AssistantMessage(
        content="Attempt a finish that will not succeed",
        tool_calls=[
            ToolCall(name="finish", arguments=finish_arguments, tool_call_id="call_finish"),
            ToolCall(name="record", arguments='{"label": "ordinary"}', tool_call_id="call_ordinary"),
        ],
        token_usage=TokenUsage(),
    )
    agent = Agent(
        client=MockLLMClient([response]),
        name="test-agent",
        tools=[record],
        finish_tool=finish,
        run_sync_in_thread=False,
    )

    _, tool_messages, finish_params = await agent.step([UserMessage(content="Test task")], {})

    assert executions == expected_executions
    assert finish_params is None
    assert [message.tool_call_id for message in tool_messages] == ["call_ordinary", "call_finish"]
    assert [message.success for message in tool_messages] == [True, False]


@pytest.mark.parametrize("arguments", ["", "   "], ids=["empty", "whitespace"])
async def test_finish_without_arguments_completes_the_run(arguments: str) -> None:
    # Clients coerce a missing arguments field to "".
    class DefaultedParams(BaseModel):
        reason: str = "task complete"

    def finish(params: DefaultedParams) -> ToolResult:
        return ToolResult(content=params.reason)

    response = AssistantMessage(
        content="Done",
        tool_calls=[ToolCall(name="finish", arguments=arguments, tool_call_id="call_finish")],
        token_usage=TokenUsage(),
    )
    agent = Agent(
        client=MockLLMClient([response]),
        name="test-agent",
        tools=[],
        finish_tool=Tool[DefaultedParams, None](
            name="finish",
            description="Finish with defaulted parameters",
            parameters=DefaultedParams,
            executor=finish,  # ty: ignore[invalid-argument-type]
        ),
        run_sync_in_thread=False,
    )

    _, tool_messages, finish_params = await agent.step([UserMessage(content="Test task")], {})

    assert finish_params == DefaultedParams(reason="task complete")
    assert [message.success for message in tool_messages] == [True]


async def test_agent_max_turns() -> None:
    """Test agent stops after max_turns is reached."""
    # Create mock responses (never calls finish)
    responses = [
        AssistantMessage(
            content=f"Turn {i}",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        )
        for i in range(5)
    ]

    # Create agent with mock client
    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=3,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    # Run agent with session context
    async with agent.session() as session:
        finish_params, _message_history, run_metadata = await session.run(
            [
                SystemMessage(content="Test system message"),
                UserMessage(content="Test task"),
            ]
        )

    # Assertions
    assert finish_params is None  # Did not finish
    assert client.call_count == 3  # Ran max_turns times
    assert isinstance(run_metadata, dict)
    # Agent's own token usage metadata should be present
    assert "token_usage" in run_metadata


async def test_context_overflow_unwinds_one_turn_and_retries() -> None:
    responses = [
        AssistantMessage(
            content="First step",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        AssistantMessage(
            content="Second step",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        ContextOverflowError("too much context"),
        AssistantMessage(
            content="Recovered",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Recovered", "paths": []}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session() as session:
        finish_params, history, _ = await session.run([UserMessage(content="Test task")])

    assert finish_params is not None
    assert finish_params.reason == "Recovered"
    assert client.call_count == 4
    assistant_contents = [msg.content for group in history for msg in group if isinstance(msg, AssistantMessage)]
    assert "First step" in assistant_contents
    assert "Second step" not in assistant_contents


async def test_context_overflow_removes_unwound_turn_metadata() -> None:
    class MarkerParams(BaseModel):
        value: str

    class MarkerMetadata(BaseModel):
        value: str

    def marker_executor(params: MarkerParams) -> ToolResult[MarkerMetadata]:
        return ToolResult(content=params.value, metadata=MarkerMetadata(value=params.value))

    marker_tool = Tool[MarkerParams, MarkerMetadata](
        name="marker",
        description="Return marker metadata",
        parameters=MarkerParams,
        executor=marker_executor,  # ty: ignore[invalid-argument-type]
    )

    responses = [
        AssistantMessage(
            content="First step",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        AssistantMessage(
            content="Second step",
            tool_calls=[ToolCall(name="marker", arguments='{"value": "dropped"}', tool_call_id="call_marker")],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        ContextOverflowError("too much context"),
        AssistantMessage(
            content="Recovered",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Recovered", "paths": []}',
                    tool_call_id="call_finish",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[marker_tool],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session() as session:
        finish_params, history, run_metadata = await session.run([UserMessage(content="Test task")])

    assert finish_params is not None
    assert "marker" not in run_metadata
    assert not any(msg.name == "marker" for group in history for msg in group if isinstance(msg, ToolMessage))


async def test_context_overflow_turn_count_uses_surviving_progress() -> None:
    responses = [
        AssistantMessage(
            content="First step",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        AssistantMessage(
            content="Second step",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        ContextOverflowError("too much context"),
        AssistantMessage(
            content="Recovered second step",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Finished after recovery", "paths": []}',
                    tool_call_id="call_finish",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=3,
        turns_remaining_warning_threshold=0,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session() as session:
        finish_params, history, _ = await session.run([UserMessage(content="Test task")])

    assistant_contents = [msg.content for group in history for msg in group if isinstance(msg, AssistantMessage)]

    assert finish_params is not None
    assert finish_params.reason == "Finished after recovery"
    assert assistant_contents == ["First step", "Recovered second step", "Done"]


async def test_cache_state_preserves_run_metadata_by_turn() -> None:
    state = CacheState(
        msgs=[UserMessage(content="Test task")],
        full_msg_history=[],
        run_metadata_by_turn={"assistant-id": {"marker": [{"value": "kept"}]}},
        task_hash="task",
    )

    data = state.to_dict()
    restored = CacheState.from_dict(data)

    assert "run_metadata" not in data
    assert restored.run_metadata_by_turn == {"assistant-id": {"marker": [{"value": "kept"}]}}


async def test_cache_state_preserves_special_user_message_types() -> None:
    state = CacheState(
        msgs=[
            SummaryMessage(content="Summary"),
            TurnWarningMessage(content="One turn left"),
            UserMessage(content="Regular user message"),
        ],
        full_msg_history=[],
        run_metadata_by_turn={},
        task_hash="task",
    )

    restored = CacheState.from_dict(state.to_dict())

    assert isinstance(restored.msgs[0], SummaryMessage)
    assert isinstance(restored.msgs[1], TurnWarningMessage)
    assert type(restored.msgs[2]) is UserMessage


async def test_context_overflow_at_original_prompt_raises() -> None:
    client = MockLLMClient([ContextOverflowError("too much context")])
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session(cache_on_interrupt=False) as session:
        with pytest.raises(ContextOverflowError, match="original prompt"):
            await session.run([UserMessage(content="Test task")])


async def test_context_overflow_does_not_unwind_first_turn_after_initial_prompt() -> None:
    responses = [
        AssistantMessage(
            content="First step",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        ContextOverflowError("too much context"),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session(cache_on_interrupt=False) as session:
        with pytest.raises(ContextOverflowError, match="original prompt"):
            await session.run([UserMessage(content="Test task")])

    assert client.call_count == 2


async def test_context_overflow_does_not_unwind_existing_summary() -> None:
    responses = [
        ContextOverflowError("first overflow"),
        ContextOverflowError("second overflow"),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session(cache_on_interrupt=False) as session:
        with pytest.raises(ContextOverflowError, match="summarized context"):
            await session.run(
                [
                    UserMessage(content="Test task"),
                    SummaryMessage(content="Summary already accepted by the model"),
                    UserMessage(content="Got it, thanks!"),
                    AssistantMessage(
                        content="Post-summary work",
                        tool_calls=[],
                        token_usage=TokenUsage(input=100, answer=50),
                    ),
                ]
            )

    assert client.call_count == 1


async def test_context_overflow_recovery_can_be_disabled() -> None:
    responses = [
        AssistantMessage(
            content="Working",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        ContextOverflowError("too much context"),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        recover_from_context_overflow=False,
    )

    async with agent.session(cache_on_interrupt=False) as session:
        with pytest.raises(ContextOverflowError, match="recovery is disabled"):
            await session.run([UserMessage(content="Test task")])


async def test_agent_tool_execution() -> None:
    """Test agent executes custom tools correctly."""

    class EchoParams(BaseModel):
        message: str

    def echo_executor(params: EchoParams) -> ToolResult:
        return ToolResult(content=f"Echo: {params.message}")

    echo_tool = Tool[EchoParams, None](
        name="echo",
        description="Echo a message",
        parameters=EchoParams,
        executor=echo_executor,  # ty: ignore[invalid-argument-type]
    )

    # Create mock responses
    responses = [
        # First turn: call echo tool
        AssistantMessage(
            content="I'll echo your message",
            tool_calls=[
                ToolCall(
                    name="echo",
                    arguments='{"message": "Hello"}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second turn: finish
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Echoed successfully", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    # Create agent with mock client
    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[echo_tool],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    # Run agent with session context
    async with agent.session() as session:
        finish_params, message_history, run_metadata = await session.run(
            [
                SystemMessage(content="Test system message"),
                UserMessage(content="Echo 'Hello'"),
            ]
        )

    # Assertions
    assert finish_params is not None
    assert finish_params.reason == "Echoed successfully"
    assert client.call_count == 2
    # Check that run metadata tracks called tools
    assert "echo" in run_metadata
    assert isinstance(run_metadata["echo"], list)
    # Agent's own token usage metadata should be present
    assert "token_usage" in run_metadata
    # Check that tool was executed
    messages = message_history[0]
    tool_messages: list[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2  # Echo tool + finish tool
    # Find the echo tool message
    echo_messages = [m for m in tool_messages if m.name == "echo"]
    assert len(echo_messages) == 1
    assert "Echo: Hello" in echo_messages[0].content


async def test_run_tool_preserves_image_content() -> None:
    """Test run_tool preserves image blocks returned by tools."""

    class EmptyParams(BaseModel):
        pass

    image_block = _sample_png_block()

    def image_executor(_params: EmptyParams) -> ToolResult:
        return ToolResult(content=[image_block])

    image_tool = Tool[EmptyParams, None](
        name="image_tool",
        description="Return an image",
        parameters=EmptyParams,
        executor=image_executor,  # ty: ignore[invalid-argument-type]
    )

    client = MockLLMClient([])
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=1,
        tools=[image_tool],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session() as session:
        tool_message = await session.run_tool(
            ToolCall(name="image_tool", arguments="{}", tool_call_id="call_1"),
            run_metadata={},
        )

    assert isinstance(tool_message.content, list)
    assert len(tool_message.content) == 1
    assert isinstance(tool_message.content[0], ImageContentBlock)
    assert tool_message.content[0].mime_type == "image/png"


async def test_agent_invalid_tool_call() -> None:
    """Test agent handles invalid tool calls gracefully."""
    # Create mock responses
    responses = [
        # Call non-existent tool
        AssistantMessage(
            content="I'll call a tool",
            tool_calls=[
                ToolCall(
                    name="nonexistent_tool",
                    arguments='{"param": "value"}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Then finish
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Handled error", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    # Create agent with mock client
    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    # Run agent with session context
    async with agent.session() as session:
        finish_params, message_history, run_metadata = await session.run(
            [
                SystemMessage(content="Test system message"),
                UserMessage(content="Test task"),
            ]
        )

    # Assertions
    assert finish_params is not None
    assert finish_params.reason == "Handled error"
    # Nonexistent tool should still be tracked (with empty metadata list)
    assert "nonexistent_tool" in run_metadata
    # Agent's own token usage metadata should be present
    assert "token_usage" in run_metadata
    # Check that tool error message was returned
    messages = message_history[0]
    tool_messages: list[ToolMessage] = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_messages) == 2  # Error message + finish tool
    # Find the error tool message
    error_messages = [m for m in tool_messages if m.name == "nonexistent_tool"]
    assert len(error_messages) == 1
    assert "not a valid tool" in error_messages[0].content


async def test_agent_finish_tool_validation() -> None:
    """Test agent only terminates on valid finish tool calls."""
    from stirrup.core.models import ToolUseCountMetadata

    class CustomFinishParams(BaseModel):
        reason: str
        status: str

    # Custom finish tool that validates status before allowing termination
    def custom_finish_executor(params: CustomFinishParams) -> ToolResult[ToolUseCountMetadata]:
        is_valid = params.status == "complete"
        return ToolResult(
            content=params.reason,
            success=is_valid,
            metadata=ToolUseCountMetadata(),
        )

    custom_finish_tool = Tool[CustomFinishParams, ToolUseCountMetadata](
        name=DEFAULT_FINISH_TOOL_NAME,
        description="Finish with status validation",
        parameters=CustomFinishParams,
        executor=custom_finish_executor,
    )

    # Create mock responses
    responses = [
        # First: invalid finish (status != "complete")
        AssistantMessage(
            content="Trying to finish",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Not ready", "status": "pending"}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second: valid finish (status == "complete")
        AssistantMessage(
            content="Now finishing",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Task done", "status": "complete"}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=custom_finish_tool,
    )

    async with agent.session() as session:
        finish_params, _, _ = await session.run([UserMessage(content="Test task")])

    # Agent should have taken 2 turns (invalid finish + valid finish)
    assert client.call_count == 2
    assert finish_params is not None
    assert finish_params.reason == "Task done"
    assert finish_params.status == "complete"


async def test_agent_accepts_multiple_finish_tools() -> None:
    """Test agent terminates when any configured finish tool succeeds."""

    class SubmitFilesParams(BaseModel):
        reason: str
        paths: list[str]

    class FinishWithoutFilesParams(BaseModel):
        reason: str

    def submit_files_executor(params: SubmitFilesParams) -> ToolResult:
        return ToolResult(content=params.reason, success=bool(params.paths))

    def finish_without_files_executor(params: FinishWithoutFilesParams) -> ToolResult:
        return ToolResult(content=params.reason, success=True)

    submit_files_tool = Tool[SubmitFilesParams, None](
        name="submit_files",
        description="Finish the task and submit created files",
        parameters=SubmitFilesParams,
        executor=submit_files_executor,  # ty: ignore[invalid-argument-type]
    )
    finish_without_files_tool = Tool[FinishWithoutFilesParams, None](
        name="finish_without_files",
        description="Finish the task without submitting files",
        parameters=FinishWithoutFilesParams,
        executor=finish_without_files_executor,  # ty: ignore[invalid-argument-type]
    )

    responses = [
        AssistantMessage(
            content="No files to submit",
            tool_calls=[
                ToolCall(
                    name="finish_without_files",
                    arguments='{"reason": "Task completed without files"}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        )
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=[submit_files_tool, finish_without_files_tool],
    )

    async with agent.session() as session:
        finish_params, _, run_metadata = await session.run([UserMessage(content="Test task")])

    assert finish_params is not None
    assert isinstance(finish_params, FinishWithoutFilesParams)
    assert finish_params.reason == "Task completed without files"
    assert client.call_count == 1
    assert set(agent.finish_tools) == {"submit_files", "finish_without_files"}
    assert set(client.tools_seen[0]) == {"submit_files", "finish_without_files"}
    assert "finish_without_files" in run_metadata


async def test_finish_tool_property_requires_single_finish_tool() -> None:
    """Test finish_tool property is only valid when one finish tool is configured."""

    class SubmitFilesParams(BaseModel):
        reason: str
        paths: list[str]

    class FinishWithoutFilesParams(BaseModel):
        reason: str

    submit_files_tool = Tool[SubmitFilesParams, None](
        name="submit_files",
        description="Finish the task and submit created files",
        parameters=SubmitFilesParams,
        executor=lambda params: ToolResult(content=params.reason),  # ty: ignore[invalid-argument-type]
    )
    finish_without_files_tool = Tool[FinishWithoutFilesParams, None](
        name="finish_without_files",
        description="Finish the task without submitting files",
        parameters=FinishWithoutFilesParams,
        executor=lambda params: ToolResult(content=params.reason),  # ty: ignore[invalid-argument-type]
    )

    agent = Agent(
        client=MockLLMClient([]),
        name="test-agent",
        tools=[],
        finish_tool=[submit_files_tool, finish_without_files_tool],
    )

    with pytest.raises(ValueError, match="multiple finish tools"):
        _ = agent.finish_tool


async def test_agent_continues_after_failed_finish_tool_from_multiple_finish_tools() -> None:
    """Test a failed finish tool call does not terminate when multiple finish tools are configured."""

    class SubmitFilesParams(BaseModel):
        reason: str
        paths: list[str]

    class FinishWithoutFilesParams(BaseModel):
        reason: str

    def submit_files_executor(params: SubmitFilesParams) -> ToolResult:
        return ToolResult(content=params.reason, success=bool(params.paths))

    def finish_without_files_executor(params: FinishWithoutFilesParams) -> ToolResult:
        return ToolResult(content=params.reason, success=True)

    submit_files_tool = Tool[SubmitFilesParams, None](
        name="submit_files",
        description="Finish the task and submit created files",
        parameters=SubmitFilesParams,
        executor=submit_files_executor,  # ty: ignore[invalid-argument-type]
    )
    finish_without_files_tool = Tool[FinishWithoutFilesParams, None](
        name="finish_without_files",
        description="Finish the task without submitting files",
        parameters=FinishWithoutFilesParams,
        executor=finish_without_files_executor,  # ty: ignore[invalid-argument-type]
    )

    responses = [
        AssistantMessage(
            content="Trying to submit without files",
            tool_calls=[
                ToolCall(
                    name="submit_files",
                    arguments='{"reason": "No files yet", "paths": []}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        AssistantMessage(
            content="Finishing without files",
            tool_calls=[
                ToolCall(
                    name="finish_without_files",
                    arguments='{"reason": "No files needed"}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=[submit_files_tool, finish_without_files_tool],
    )

    async with agent.session() as session:
        finish_params, history, _ = await session.run([UserMessage(content="Test task")])

    assert client.call_count == 2
    assert finish_params is not None
    assert isinstance(finish_params, FinishWithoutFilesParams)
    assert finish_params.reason == "No files needed"

    tool_messages = [msg for group in history for msg in group if isinstance(msg, ToolMessage)]
    assert len([msg for msg in tool_messages if msg.name == "submit_files" and not msg.success]) == 1


async def test_multiple_finish_tool_calls_in_one_turn_all_rejected() -> None:
    """When multiple finish tools are called in one turn, all are rejected without executing.

    The ordinary calls in that turn still run, and the agent must not terminate; the model gets
    another chance to finish with a single call.
    """
    executions: list[str] = []
    record = recording_tool("record", executions)

    class FinishAParams(BaseModel):
        reason: str

    class FinishBParams(BaseModel):
        reason: str

    def finish_a_executor(params: FinishAParams) -> ToolResult:
        executions.append("finish_a")
        return ToolResult(content=params.reason, success=True)

    def finish_b_executor(params: FinishBParams) -> ToolResult:
        executions.append("finish_b")
        return ToolResult(content=params.reason, success=True)

    finish_a = Tool[FinishAParams, None](
        name="finish_a",
        description="Finish variant A",
        parameters=FinishAParams,
        executor=finish_a_executor,  # ty: ignore[invalid-argument-type]
    )
    finish_b = Tool[FinishBParams, None](
        name="finish_b",
        description="Finish variant B",
        parameters=FinishBParams,
        executor=finish_b_executor,  # ty: ignore[invalid-argument-type]
    )

    responses = [
        AssistantMessage(
            content="Calling both finish tools at once",
            tool_calls=[
                ToolCall(name="finish_a", arguments='{"reason": "first"}', tool_call_id="call_1"),
                ToolCall(name="finish_b", arguments='{"reason": "second"}', tool_call_id="call_2"),
                ToolCall(name="record", arguments='{"label": "ordinary"}', tool_call_id="call_ordinary"),
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        AssistantMessage(
            content="Retrying with a single finish call",
            tool_calls=[
                ToolCall(name="finish_a", arguments='{"reason": "settled"}', tool_call_id="call_3"),
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[record],
        finish_tool=[finish_a, finish_b],
    )

    async with agent.session() as session:
        finish_params, history, _ = await session.run([UserMessage(content="Test task")])

    # Neither finish tool was executed in turn 1, but the ordinary call still ran.
    assert executions == ["ordinary", "finish_a"]
    assert client.call_count == 2
    assert finish_params is not None
    assert isinstance(finish_params, FinishAParams)
    assert finish_params.reason == "settled"

    tool_messages = [msg for group in history for msg in group if isinstance(msg, ToolMessage)]
    rejected = [msg for msg in tool_messages if not msg.success and "multiple finish tools" in str(msg.content)]
    assert {msg.name for msg in rejected} == {"finish_a", "finish_b"}


async def test_tools_finish_tool_name_collision_raises_at_init() -> None:
    """Agent init raises if a tool in `tools` shares a name with a finish tool."""

    class DummyParams(BaseModel):
        x: str

    colliding_tool = Tool[DummyParams, None](
        name="finish_a",
        description="Not actually a finish tool",
        parameters=DummyParams,
        executor=lambda params: ToolResult(content=params.x),  # ty: ignore[invalid-argument-type]
    )
    finish_a = Tool[DummyParams, None](
        name="finish_a",
        description="Real finish tool",
        parameters=DummyParams,
        executor=lambda params: ToolResult(content=params.x, success=True),  # ty: ignore[invalid-argument-type]
    )

    with pytest.raises(ValueError, match="collides with a finish tool"):
        Agent(
            client=MockLLMClient([]),
            name="test-agent",
            tools=[colliding_tool],
            finish_tool=finish_a,
        )


async def test_finish_tool_validates_file_paths() -> None:
    """Test that SIMPLE_FINISH_TOOL rejects non-existent file paths."""
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    # Create mock responses
    responses = [
        # First: finish with non-existent file path
        AssistantMessage(
            content="Finishing with fake file",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Done", "paths": ["nonexistent.txt"]}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second: finish with empty paths (should succeed)
        AssistantMessage(
            content="Finishing properly",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Actually done", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[LocalCodeExecToolProvider()],
    )

    async with agent.session() as session:
        finish_params, history, _ = await session.run([UserMessage(content="Test task")])

    # Agent should have taken 2 turns (failed finish + successful finish)
    assert client.call_count == 2
    assert finish_params is not None
    assert finish_params.reason == "Actually done"

    # First finish should have failed with error about missing file
    tool_messages = [msg for group in history for msg in group if isinstance(msg, ToolMessage)]
    assert any("nonexistent.txt" in str(msg.content) and not msg.success for msg in tool_messages)


async def test_no_successive_assistant_messages() -> None:
    """Test agent adds continue message to avoid successive assistant messages."""
    responses = [
        # First: assistant message without tool calls
        AssistantMessage(
            content="Let me think about this",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second: finish after continue
        AssistantMessage(
            content="Now I'll finish",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Task completed", "paths": []}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=30,  # Use default max_turns so warning threshold won't be hit
        turns_remaining_warning_threshold=5,  # Only warn in last 5 turns
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session() as session:
        finish_params, message_history, _ = await session.run([UserMessage(content="Test task")])

    # Verify finish params
    assert finish_params is not None
    assert finish_params.reason == "Task completed"
    assert client.call_count == 2

    # Verify "Please continue the task" message was added after first assistant message
    messages = message_history[0]
    continue_messages = [m for m in messages if isinstance(m, UserMessage) and m.content == "Please continue the task"]
    assert len(continue_messages) == 1


async def test_allow_successive_assistant_messages() -> None:
    """Test agent allows successive assistant messages when flag is enabled."""
    responses = [
        # First: assistant message without tool calls
        AssistantMessage(
            content="Let me think about this",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        # Second: another assistant message without continue prompt
        AssistantMessage(
            content="Now I'll finish",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Task completed", "paths": []}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=30,
        turns_remaining_warning_threshold=5,
        block_successive_assistant_messages=False,  # Disable blocking
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with agent.session() as session:
        finish_params, message_history, _ = await session.run([UserMessage(content="Test task")])

    # Verify finish params
    assert finish_params is not None
    assert finish_params.reason == "Task completed"
    assert client.call_count == 2

    # Verify NO "Please continue the task" message was added
    messages = message_history[0]
    continue_messages = [m for m in messages if isinstance(m, UserMessage) and m.content == "Please continue the task"]
    assert len(continue_messages) == 0


async def test_summarize_history_has_one_summary_per_trajectory() -> None:
    """Test that each sub-trajectory in history contains at most one SummaryMessage.

    Simulates an agent run where summarization triggers twice. Verifies:
    - history[0] (pre-first-summary) has 0 SummaryMessages
    - history[1] (post-first-summary) has exactly 1 SummaryMessage
    - history[2] (post-second-summary, final) has exactly 1 SummaryMessage
    """
    # max_tokens=1000 and cutoff=0.3 means summarization triggers when
    # token_usage.total >= 300. Turns without tool calls also trigger
    # "Please continue" messages from block_successive_assistant_messages.

    responses = [
        # Turn 1: high token usage triggers first summarization
        AssistantMessage(
            content="Working on it",
            tool_calls=[],
            token_usage=TokenUsage(input=250, answer=100),  # total=350 >= 300
        ),
        # First summarization generate call
        AssistantMessage(
            content="First summary of progress.",
            tool_calls=[],
            token_usage=TokenUsage(input=200, answer=50),
        ),
        # Turn 2: high token usage triggers second summarization
        AssistantMessage(
            content="Continuing work",
            tool_calls=[],
            token_usage=TokenUsage(input=250, answer=100),  # total=350 >= 300
        ),
        # Second summarization generate call
        AssistantMessage(
            content="Second summary of progress.",
            tool_calls=[],
            token_usage=TokenUsage(input=200, answer=50),
        ),
        # Turn 3: finish
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Completed", "paths": []}',
                    tool_call_id="call_finish",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses, max_tokens=1000)

    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=10,
        turns_remaining_warning_threshold=2,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        context_summarization_cutoff=0.3,
    )

    async with agent.session() as session:
        _finish_params, history, _ = await session.run(
            [SystemMessage(content="System prompt"), UserMessage(content="Do the task")]
        )

    # Should have 3 sub-trajectories: pre-summary, post-1st-summary, post-2nd-summary (final)
    assert len(history) == 3

    # history[0]: original conversation before first summarization — no summaries
    summaries_0 = [m for m in history[0] if isinstance(m, SummaryMessage)]
    assert len(summaries_0) == 0

    # history[1]: after first summarization — exactly 1 SummaryMessage
    summaries_1 = [m for m in history[1] if isinstance(m, SummaryMessage)]
    assert len(summaries_1) == 1

    # history[2]: after second summarization — exactly 1 SummaryMessage (not 2)
    summaries_2 = [m for m in history[2] if isinstance(m, SummaryMessage)]
    assert len(summaries_2) == 1

    # The summary content should be different between history[1] and history[2]
    assert summaries_1[0].content != summaries_2[0].content


async def test_summarization_context_overflow_unwinds_and_retries() -> None:
    responses = [
        AssistantMessage(
            content="First step",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        AssistantMessage(
            content="Second step",
            tool_calls=[],
            token_usage=TokenUsage(input=250, answer=100),
        ),
        ContextOverflowError("summary context overflow"),
        AssistantMessage(
            content="Recovered summary.",
            tool_calls=[],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "Completed", "paths": []}',
                    tool_call_id="call_finish",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    client = MockLLMClient(responses, max_tokens=1000)
    agent = Agent(
        client=client,
        name="test-agent",
        max_turns=5,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        context_summarization_cutoff=0.3,
    )

    async with agent.session() as session:
        finish_params, history, _ = await session.run(
            [SystemMessage(content="System prompt"), UserMessage(content="Do the task")]
        )

    assert finish_params is not None
    assert finish_params.reason == "Completed"
    assert client.call_count == 5
    assert any(isinstance(msg, SummaryMessage) for msg in history[-1])


async def test_shared_subagent_normalizes_absolute_local_output_and_rejects_directory() -> None:
    """A shared sub-agent reports a usable relative path and never endorses a directory."""
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    subagent_client = MockLLMClient([])
    subagent = Agent(
        client=subagent_client,
        name="shared-worker",
        tools=[LocalCodeExecToolProvider()],
        finish_tool=UNVALIDATED_FINISH_TOOL,
        share_parent_exec_env=True,
    )
    subagent_tool = subagent.to_tool()
    parent_provider = LocalCodeExecToolProvider()
    parent = Agent(
        client=MockLLMClient([]),
        name="parent",
        tools=[parent_provider, subagent_tool],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with parent.session():
        await parent_provider.write_file_bytes("report.txt", b"report")
        assert parent_provider.temp_dir is not None
        (parent_provider.temp_dir / "artifacts").mkdir()
        absolute_report = str(parent_provider.temp_dir / "report.txt")
        subagent_client.responses.append(
            AssistantMessage(
                content="Done",
                tool_calls=[
                    ToolCall(
                        name=DEFAULT_FINISH_TOOL_NAME,
                        arguments=json.dumps({"reason": "done", "paths": [absolute_report, "artifacts"]}),
                        tool_call_id="call_finish",
                    )
                ],
                token_usage=TokenUsage(input=100, answer=50),
            )
        )
        execution = subagent_tool.executor(SubAgentParams(task="produce outputs", input_files=[]))
        assert isinstance(execution, Awaitable)
        result = await execution

    assert isinstance(result.content, str)
    assert "Files available in your environment: ['report.txt']" in result.content
    assert absolute_report not in result.content.split("Files available in your environment:", 1)[1]
    failed_outputs = result.content.split("Files that FAILED to transfer to your environment", 1)[1]
    assert "'artifacts':" in failed_outputs


@pytest.mark.docker
async def test_shared_docker_subagent_repairs_and_validates_absolute_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shared Docker outputs are repaired, normalized, and still constrained to regular files."""
    pytest.importorskip("docker")

    from stirrup.tools.code_backends.base import CommandResult
    from stirrup.tools.code_backends.docker import DEFAULT_WORKING_DIR, DockerCodeExecToolProvider

    mock_docker_client = MagicMock()
    mock_container = MagicMock()
    mock_container.short_id = "abc123"
    mock_docker_client.containers.run.return_value = mock_container

    declared_file = f"{DEFAULT_WORKING_DIR}/root-owned/report.txt"
    declared_directory = f"{DEFAULT_WORKING_DIR}/artifacts"
    traversal = f"{DEFAULT_WORKING_DIR}/nested/../escape.txt"
    subagent = Agent(
        client=MockLLMClient(
            [
                AssistantMessage(
                    content="Done",
                    tool_calls=[
                        ToolCall(
                            name=DEFAULT_FINISH_TOOL_NAME,
                            arguments=json.dumps(
                                {"reason": "done", "paths": [declared_file, declared_directory, traversal]}
                            ),
                            tool_call_id="call_finish",
                        )
                    ],
                    token_usage=TokenUsage(input=100, answer=50),
                )
            ]
        ),
        name="shared-docker-worker",
        tools=[DockerCodeExecToolProvider.from_image("python:3.12-slim")],
        finish_tool=UNVALIDATED_FINISH_TOOL,
        share_parent_exec_env=True,
    )
    subagent_tool = subagent.to_tool()
    parent_provider = DockerCodeExecToolProvider.from_image("python:3.12-slim", temp_base_dir=tmp_path)
    parent = Agent(
        client=MockLLMClient([]),
        name="docker-parent",
        tools=[parent_provider, subagent_tool],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    with (
        patch("stirrup.tools.code_backends.docker.docker.from_env", return_value=mock_docker_client),
        patch("stirrup.tools.code_backends.docker.to_thread") as mock_to_thread,
    ):
        mock_to_thread.run_sync = AsyncMock(side_effect=lambda fn, *args: fn(*args) if not args else fn)

        async with parent.session():
            assert parent_provider.temp_dir is not None
            locked_parent = parent_provider.temp_dir / "root-owned"
            locked_parent.mkdir()
            (locked_parent / "report.txt").write_bytes(b"report")
            locked_parent.chmod(0)
            (parent_provider.temp_dir / "artifacts").mkdir()

            async def repair_access(cmd: str, *, timeout: int | None = None) -> CommandResult:  # noqa: ARG001
                locked_parent.chmod(0o700)
                return CommandResult(exit_code=0, stdout="", stderr="")

            monkeypatch.setattr(parent_provider, "_run_command_unchecked", repair_access)
            execution = subagent_tool.executor(SubAgentParams(task="produce outputs", input_files=[]))
            assert isinstance(execution, Awaitable)
            result = await execution

    assert isinstance(result.content, str)
    available_outputs = result.content.split("Files available in your environment:", 1)[1]
    assert "['root-owned/report.txt']" in available_outputs
    assert declared_file not in available_outputs
    failed_outputs = result.content.split("Files that FAILED to transfer to your environment", 1)[1]
    assert f"'{declared_directory}':" in failed_outputs
    assert f"'{traversal}': 'Output source path contains traversal" in result.content


async def test_owned_subagent_without_parent_reports_declared_output_failure() -> None:
    """A directly invoked owned sub-agent cannot silently drop declared outputs."""
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    responses = [
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "done", "paths": ["report.txt"]}',
                    tool_call_id="call_finish",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]
    subagent = Agent(
        client=MockLLMClient(responses),
        name="owned-worker",
        tools=[LocalCodeExecToolProvider()],
        finish_tool=UNVALIDATED_FINISH_TOOL,
    )

    execution = subagent.to_tool().executor(SubAgentParams(task="produce output", input_files=[]))
    assert isinstance(execution, Awaitable)
    result = await execution

    assert "Files available in your environment" not in result.content
    assert "Files that FAILED to transfer to your environment" in result.content
    assert "No parent execution environment is available for transfer" in result.content


async def test_root_agent_saves_declared_outputs_and_records_failures(tmp_path: Path) -> None:
    """Output transfer runs at session exit: saved files land in output_dir and
    failures are recorded on last_output_files_result, not silently dropped.

    The failing path exists at finish time (so the finish tool accepts it) but
    uses a traversal spelling that the transfer rejects — the seam where
    failure surfacing actually matters.
    """
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    responses = [
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "done", "paths": ["real.txt", "nested/../sneaky.txt"]}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
            request_start_time=100.0,
            request_end_time=100.4,
        )
    ]

    provider = LocalCodeExecToolProvider()
    agent = Agent(
        client=MockLLMClient(responses),
        name="test-agent",
        max_turns=3,
        tools=[provider],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    output_dir = tmp_path / "outputs"
    async with agent.session(output_dir=output_dir) as session:
        await provider.write_file_bytes("real.txt", b"data")
        await provider.write_file_bytes("sneaky.txt", b"sneaky")
        await provider.write_file_bytes("nested/anchor.txt", b"")
        finish_params, _, _ = await session.run([SystemMessage(content="sys"), UserMessage(content="task")])

    assert finish_params is not None
    assert (output_dir / "real.txt").read_bytes() == b"data"
    assert not (output_dir / "sneaky.txt").exists()

    result = session.last_output_files_result
    assert result is not None
    assert [saved.source_path for saved in result.saved] == ["real.txt"]
    assert "traversal" in result.failed["nested/../sneaky.txt"]


async def test_root_session_resets_stale_output_result(tmp_path: Path) -> None:
    """A later root run with no declared outputs must not expose the previous
    run's last_output_files_result."""
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    responses = [
        AssistantMessage(
            content="first",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "done", "paths": ["real.txt"]}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        AssistantMessage(
            content="second",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "done", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
    ]

    provider = LocalCodeExecToolProvider()
    agent = Agent(
        client=MockLLMClient(responses),
        name="test-agent",
        max_turns=3,
        tools=[provider],
        finish_tool=SIMPLE_FINISH_TOOL,
    )
    output_dir = tmp_path / "outputs"

    async with agent.session(output_dir=output_dir) as session:
        await provider.write_file_bytes("real.txt", b"data")
        await session.run([SystemMessage(content="sys"), UserMessage(content="task")])
    completed_session = session
    completed_result = session.last_output_files_result
    assert completed_result is not None

    async with agent.session(output_dir=output_dir) as session:
        await session.run([SystemMessage(content="sys"), UserMessage(content="task")])
    assert session.last_output_files_result is None
    assert completed_session.last_output_files_result is completed_result


async def test_failed_reused_session_does_not_export_previous_finish_paths(tmp_path: Path) -> None:
    """Only finish parameters produced by the current session may drive output saving."""
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    responses: list[AssistantMessage | Exception] = [
        AssistantMessage(
            content="first",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "done", "paths": ["report.txt"]}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        RuntimeError("second session failed before finish"),
    ]
    provider = LocalCodeExecToolProvider()
    agent = Agent(
        client=MockLLMClient(responses),
        name="test-agent",
        max_turns=3,
        tools=[provider],
        finish_tool=SIMPLE_FINISH_TOOL,
    )
    output_dir = tmp_path / "outputs"

    async with agent.session(output_dir=output_dir, cache_on_interrupt=False) as session:
        await provider.write_file_bytes("report.txt", b"first")
        await session.run([SystemMessage(content="sys"), UserMessage(content="first")])

    with pytest.raises(RuntimeError, match="failed before finish"):
        async with agent.session(output_dir=output_dir, cache_on_interrupt=False) as failed_session:
            await provider.write_file_bytes("report.txt", b"second")
            await failed_session.run([SystemMessage(content="sys"), UserMessage(content="second")])

    assert (output_dir / "report.txt").read_bytes() == b"first"
    assert failed_session.last_output_files_result is None


async def test_failed_second_run_in_same_session_does_not_reuse_first_finish(tmp_path: Path) -> None:
    """Each run must replace the session's pending finish state before doing work."""
    from stirrup.tools.code_backends.local import LocalCodeExecToolProvider

    responses: list[AssistantMessage | Exception] = [
        AssistantMessage(
            content="first",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments='{"reason": "done", "paths": ["report.txt"]}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, answer=50),
        ),
        RuntimeError("second run failed before finish"),
    ]
    provider = LocalCodeExecToolProvider()
    agent = Agent(
        client=MockLLMClient(responses),
        name="test-agent",
        max_turns=3,
        tools=[provider],
        finish_tool=SIMPLE_FINISH_TOOL,
    )
    output_dir = tmp_path / "outputs"

    with pytest.raises(RuntimeError, match="failed before finish"):
        async with agent.session(output_dir=output_dir, cache_on_interrupt=False) as session:
            await provider.write_file_bytes("report.txt", b"data")
            await session.run([SystemMessage(content="sys"), UserMessage(content="first")])
            await session.run([SystemMessage(content="sys"), UserMessage(content="second")])

    assert not (output_dir / "report.txt").exists()
    assert session.last_output_files_result is None


def test_default_tools_returns_fresh_instances() -> None:
    """Each call must build new providers so concurrent sessions never share one."""
    first = default_tools()
    second = default_tools()

    assert [type(tool) for tool in first] == [type(tool) for tool in second]
    assert all(a is not b for a, b in zip(first, second, strict=True))


def test_agents_do_not_share_default_tool_instances() -> None:
    """The default path must not hand two Agents the same providers.

    Providers hold per-session state, so a shared instance lets one session's teardown
    destroy another's execution environment.
    """
    first = Agent(client=MockLLMClient([]), name="first")
    second = Agent(client=MockLLMClient([]), name="second")

    assert [type(tool) for tool in first._tools] == [type(tool) for tool in second._tools]  # noqa: SLF001
    assert all(a is not b for a, b in zip(first._tools, second._tools, strict=True))  # noqa: SLF001
