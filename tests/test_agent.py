"""Tests for agent core functionality."""

from pydantic import BaseModel

from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core.agent import Agent
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    LLMClient,
    SummaryMessage,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from stirrup.tools.finish import SIMPLE_FINISH_TOOL, FinishParams


class MockLLMClient(LLMClient):
    """Mock LLM client for testing."""

    def __init__(self, responses: list[AssistantMessage], max_tokens: int = 100_000) -> None:
        self.responses = responses
        self.call_count = 0
        self._max_tokens = max_tokens
        self.tools_seen: list[dict[str, Tool]] = []

    @property
    def model_slug(self) -> str:
        return "mock-model"

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:  # noqa: ARG002
        self.tools_seen.append(tools)
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


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

    try:
        _ = agent.finish_tool
    except ValueError as exc:
        assert "multiple finish tools" in str(exc)
    else:
        raise AssertionError("finish_tool should fail when multiple finish tools are configured")


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

    The agent must not terminate; the model gets another chance to finish with a single call.
    """
    executions: list[str] = []

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
        tools=[],
        finish_tool=[finish_a, finish_b],
    )

    async with agent.session() as session:
        finish_params, history, _ = await session.run([UserMessage(content="Test task")])

    # Neither finish tool was executed in turn 1 — agent recovered on turn 2.
    assert executions == ["finish_a"]
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

    try:
        Agent(
            client=MockLLMClient([]),
            name="test-agent",
            tools=[colliding_tool],
            finish_tool=finish_a,
        )
    except ValueError as exc:
        assert "collides with a finish tool" in str(exc)
    else:
        raise AssertionError("Agent init should fail when a tool name collides with a finish tool")


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
