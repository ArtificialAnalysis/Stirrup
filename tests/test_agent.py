"""Tests for agent core functionality."""

from pydantic import BaseModel

from stirrup.constants import FINISH_TOOL_NAME
from stirrup.core.agent import Agent
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    LLMClient,
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

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = responses
        self.call_count = 0

    @property
    def model_slug(self) -> str:
        return "mock-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:  # noqa: ARG002
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
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Task completed successfully", "paths": []}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, output=50),
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
            token_usage=TokenUsage(input=100, output=50),
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
            token_usage=TokenUsage(input=100, output=50),
        ),
        # Second turn: finish
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Echoed successfully", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, output=50),
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
            token_usage=TokenUsage(input=100, output=50),
        ),
        # Then finish
        AssistantMessage(
            content="Done",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Handled error", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, output=50),
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
        name=FINISH_TOOL_NAME,
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
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Not ready", "status": "pending"}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, output=50),
        ),
        # Second: valid finish (status == "complete")
        AssistantMessage(
            content="Now finishing",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Task done", "status": "complete"}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, output=50),
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
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Done", "paths": ["nonexistent.txt"]}',
                    tool_call_id="call_1",
                )
            ],
            token_usage=TokenUsage(input=100, output=50),
        ),
        # Second: finish with empty paths (should succeed)
        AssistantMessage(
            content="Finishing properly",
            tool_calls=[
                ToolCall(
                    name=FINISH_TOOL_NAME,
                    arguments='{"reason": "Actually done", "paths": []}',
                    tool_call_id="call_2",
                )
            ],
            token_usage=TokenUsage(input=100, output=50),
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
