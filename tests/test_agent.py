"""Tests for agent core functionality."""

from pathlib import Path

import pytest
from pydantic import BaseModel

from stirrup.constants import FINISH_TOOL_NAME
from stirrup.core import cache as cache_module
from stirrup.core.agent import Agent
from stirrup.core.cache import CacheManager, compute_task_hash
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    EmptyMetadata,
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


class MockLLMClient(LLMClient[EmptyMetadata]):
    """Mock LLM client for testing."""

    assistant_metadata_type: type[EmptyMetadata] = EmptyMetadata

    def __init__(self, responses: list[AssistantMessage[EmptyMetadata]], max_tokens: int = 100_000) -> None:
        self.responses = responses
        self.call_count = 0
        self._max_tokens = max_tokens

    @property
    def model_slug(self) -> str:
        return "mock-model"

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    async def generate(
        self,
        messages: list[ChatMessage[EmptyMetadata]],
        tools: dict[str, Tool],
    ) -> AssistantMessage[EmptyMetadata]:  # noqa: ARG002
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
                    name=FINISH_TOOL_NAME,
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
                    name=FINISH_TOOL_NAME,
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
            token_usage=TokenUsage(input=100, answer=50),
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
            token_usage=TokenUsage(input=100, answer=50),
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
                    name=FINISH_TOOL_NAME,
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
                    name=FINISH_TOOL_NAME,
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
                    name=FINISH_TOOL_NAME,
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


async def test_agent_resume_loads_cached_state_and_clears_cache_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test cache save/load and resume flow for an interrupted run."""
    # Use a temporary cache directory
    monkeypatch.setattr(cache_module, "DEFAULT_CACHE_DIR", tmp_path)

    init_msgs = [UserMessage(content="Test task")]
    task_hash = compute_task_hash(init_msgs)
    cache_manager = CacheManager(cache_base_dir=tmp_path)

    # Create an unfinished run that should be cached on exit
    first_client = MockLLMClient(
        [
            AssistantMessage(
                content="Still working",
                tool_calls=[],
                token_usage=TokenUsage(input=100, answer=50),
            )
        ]
    )
    first_agent = Agent(
        client=first_client,
        name="test-agent",
        max_turns=1,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    # Run once without finishing to create the cache
    async with first_agent.session(cache_on_interrupt=False) as session:
        finish_params, _, _ = await session.run(init_msgs)

    # Verify cache was written with the pre-step state
    assert finish_params is None
    assert first_client.call_count == 1
    assert (tmp_path / task_hash / "state.json").exists()

    cached = cache_manager.load_state(task_hash, EmptyMetadata)
    assert cached is not None
    assert cached.turn == 0
    assert cached.full_msg_history == []
    assert len(cached.msgs) == 2
    assert isinstance(cached.msgs[0], SystemMessage)
    assert isinstance(cached.msgs[1], UserMessage)
    assert cached.msgs[1].content == "Test task"

    # Resume the same task and finish successfully
    second_client = MockLLMClient(
        [
            AssistantMessage(
                content="Done",
                tool_calls=[
                    ToolCall(
                        name=FINISH_TOOL_NAME,
                        arguments='{"reason": "Resumed successfully", "paths": []}',
                        tool_call_id="call_1",
                    )
                ],
                token_usage=TokenUsage(input=100, answer=50),
            )
        ]
    )
    second_agent = Agent(
        client=second_client,
        name="test-agent",
        max_turns=1,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
    )

    async with second_agent.session(resume=True, cache_on_interrupt=False) as session:
        finish_params, history, _ = await session.run(init_msgs)

    # Verify the resumed run completed and cleared the cache
    assert finish_params is not None
    assert finish_params.reason == "Resumed successfully"
    assert second_client.call_count == 1
    assert len(history) == 1
    assert cache_manager.load_state(task_hash, EmptyMetadata) is None
