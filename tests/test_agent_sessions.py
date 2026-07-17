"""Behavioral contracts for isolated Agent sessions."""

import json
import signal
from io import StringIO
from pathlib import Path
from typing import Any, Self

import anyio
import pytest
from PIL import Image
from pydantic import BaseModel
from rich.console import Console

from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core.agent import Agent
from stirrup.core.cache import CacheManager, CacheState, compute_task_hash
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    LLMClient,
    SubAgentMetadata,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolProvider,
    ToolResult,
    UserMessage,
)
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL
from stirrup.tools.view_image import ViewImageToolProvider
from stirrup.utils.logging import AgentLogger, AgentLoggerBase


class ConcurrentPromptClient(LLMClient):
    """Finish two overlapping runs from the prompt visible to each call."""

    def __init__(self) -> None:
        self._arrived = {"first prompt": anyio.Event(), "second prompt": anyio.Event()}

    @property
    def model_slug(self) -> str:
        return "concurrent-prompt-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        assert DEFAULT_FINISH_TOOL_NAME in tools
        prompt = next(
            str(message.content)
            for message in reversed(messages)
            if isinstance(message, UserMessage) and str(message.content) in self._arrived
        )
        self._arrived[prompt].set()
        other_prompt = "second prompt" if prompt == "first prompt" else "first prompt"
        await self._arrived[other_prompt].wait()
        return _finish_response(prompt, f"finish-{prompt}")


class ConcurrentIdenticalClient(LLMClient):
    """Finish only after two calls for the same prompt overlap."""

    def __init__(self) -> None:
        self.call_count = 0
        self.all_arrived = anyio.Event()

    @property
    def model_slug(self) -> str:
        return "concurrent-identical-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        del messages
        assert DEFAULT_FINISH_TOOL_NAME in tools
        self.call_count += 1
        if self.call_count == 2:
            self.all_arrived.set()
        await self.all_arrived.wait()
        return _finish_response("identical", f"finish-identical-{self.call_count}")


class BlockingFinishClient(LLMClient):
    """Hold a run active until its cache boundary has been inspected."""

    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()

    @property
    def model_slug(self) -> str:
        return "blocking-finish-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        del messages
        assert DEFAULT_FINISH_TOOL_NAME in tools
        self.started.set()
        await self.release.wait()
        return _finish_response("finished", "finish-blocked")


class ConcurrentFileClient(LLMClient):
    """Copy each overlapping session's input into its configured output."""

    def __init__(self) -> None:
        self._arrived = {"first file": anyio.Event(), "second file": anyio.Event()}

    @property
    def model_slug(self) -> str:
        return "concurrent-file-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        assert "code_exec" in tools
        prompt = next(
            str(message.content)
            for message in reversed(messages)
            if isinstance(message, UserMessage) and str(message.content) in self._arrived
        )
        if not any(isinstance(message, ToolMessage) and message.name == "code_exec" for message in messages):
            self._arrived[prompt].set()
            other_prompt = "second file" if prompt == "first file" else "first file"
            await self._arrived[other_prompt].wait()
            return AssistantMessage(
                content=f"copying {prompt}",
                tool_calls=[
                    ToolCall(
                        name="code_exec",
                        arguments='{"cmd": "cp input.txt result.txt"}',
                        tool_call_id=f"copy-{prompt}",
                    )
                ],
                token_usage=TokenUsage(),
            )
        return AssistantMessage(
            content=f"finished {prompt}",
            tool_calls=[
                ToolCall(
                    name=DEFAULT_FINISH_TOOL_NAME,
                    arguments=json.dumps({"reason": prompt, "paths": ["result.txt"]}),
                    tool_call_id=f"finish-{prompt}",
                )
            ],
            token_usage=TokenUsage(),
        )


class ScriptedClient(LLMClient):
    def __init__(self, responses: list[AssistantMessage | Exception]) -> None:
        self.responses = responses
        self.messages_seen: list[list[ChatMessage]] = []

    @property
    def model_slug(self) -> str:
        return "scripted-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        del tools
        self.messages_seen.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class CustomProvider(ToolProvider):
    """Stateful custom provider used to verify sequential-only ownership."""

    def __init__(self) -> None:
        self.enter_count = 0
        self.exit_count = 0
        self.is_active = False

    async def __aenter__(self) -> Tool:
        self.enter_count += 1
        self.is_active = True

        def inspect(_params: BaseModel) -> ToolResult:
            return ToolResult(content=str(self.enter_count), success=self.is_active)

        return Tool(name="inspect_custom", description="Inspect custom state", executor=inspect)

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.exit_count += 1
        self.is_active = False


class BlockingCleanupProvider(CustomProvider):
    """Custom provider whose asynchronous cleanup is externally released."""

    def __init__(self) -> None:
        super().__init__()
        self.cleanup_started = anyio.Event()
        self.cleanup_allowed = anyio.Event()
        self.cleanup_finished = anyio.Event()
        self.first_exit_exception: BaseException | None = None
        self.first_cleanup_error = RuntimeError("cancelled cleanup failed")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        if self.enter_count == 1:
            self.first_exit_exception = exc_val
        self.cleanup_started.set()
        await self.cleanup_allowed.wait()
        await super().__aexit__(exc_type, exc_val, exc_tb)
        self.cleanup_finished.set()
        if self.exit_count == 1:
            raise self.first_cleanup_error


class CleanupFailingProvider(CustomProvider):
    """Custom provider that records cleanup before failing it."""

    def __init__(self) -> None:
        super().__init__()
        self.exit_error = RuntimeError("provider cleanup failed")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await super().__aexit__(exc_type, exc_val, exc_tb)
        raise self.exit_error


class CountingViewImageProvider(ViewImageToolProvider):
    """Custom view-image provider used to verify dependent backend ownership."""

    def __init__(self, exec_env: LocalCodeExecToolProvider) -> None:
        super().__init__(exec_env)
        self.enter_count = 0

    async def __aenter__(self) -> Tool:
        self.enter_count += 1
        return await super().__aenter__()


class CustomLogger(AgentLoggerBase):
    """Minimal stateful logger used to verify sequential-only ownership."""

    def __init__(self) -> None:
        self.name = "agent"
        self.model: str | None = None
        self.max_turns: int | None = None
        self.depth = 0
        self.finish_params: BaseModel | None = None
        self.run_metadata: dict[str, list[Any]] | None = None
        self.output_dir: str | None = None
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self) -> Self:
        self.enter_count += 1
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        del exc_type, exc_val, exc_tb
        self.exit_count += 1

    def on_step(self, step: int, tool_calls: int = 0, input_tokens: int = 0, output_tokens: int = 0) -> None:
        del step, tool_calls, input_tokens, output_tokens

    def assistant_message(self, turn: int, max_turns: int, assistant_message: AssistantMessage) -> None:
        del turn, max_turns, assistant_message

    def user_message(self, user_message: UserMessage) -> None:
        del user_message

    def task_message(self, task: str | list[Any]) -> None:
        del task

    def tool_result(self, tool_message: ToolMessage) -> None:
        del tool_message

    def context_summarization_start(self, pct_used: float, cutoff: float) -> None:
        del pct_used, cutoff

    def context_summarization_complete(self, summary: str, bridge: str) -> None:
        del summary, bridge

    def debug(self, message: str, *args: object) -> None:
        del message, args

    def info(self, message: str, *args: object) -> None:
        del message, args

    def warning(self, message: str, *args: object) -> None:
        del message, args

    def error(self, message: str, *args: object) -> None:
        del message, args


class CleanupFailingLogger(CustomLogger):
    """Custom logger that records cleanup before failing it."""

    def __init__(self) -> None:
        super().__init__()
        self.exit_error = RuntimeError("logger cleanup failed")

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: object) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        raise self.exit_error


class WarningRecordingLogger(AgentLogger):
    """Built-in logger subclass that records warnings without rendering them."""

    def __init__(self) -> None:
        super().__init__(show_spinner=False)
        self.warnings_seen: list[str] = []

    def warnings_message(self, warnings: list[str]) -> None:
        self.warnings_seen.extend(warnings)


def _finish_response(reason: str, tool_call_id: str) -> AssistantMessage:
    return AssistantMessage(
        content=f"finished: {reason}",
        tool_calls=[
            ToolCall(
                name=DEFAULT_FINISH_TOOL_NAME,
                arguments=json.dumps({"reason": reason, "paths": []}),
                tool_call_id=tool_call_id,
            )
        ],
        token_usage=TokenUsage(),
    )


def _write_skill(skills_dir: Path, name: str) -> None:
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Available to delegated work.\n---\n\n# {name}\n"
    )


def _quiet_logger() -> AgentLogger:
    return AgentLogger(show_spinner=False)


async def test_concurrent_exact_builtin_sessions_isolate_inputs_outputs_and_run_state(tmp_path: Path) -> None:
    agent = Agent(
        client=ConcurrentFileClient(),
        name="file-agent",
        max_turns=2,
        tools=[LocalCodeExecToolProvider()],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    inputs: dict[str, Path] = {}
    outputs: dict[str, Path] = {}
    for prompt, content in (("first file", "FIRST-DATA"), ("second file", "SECOND-DATA")):
        input_path = tmp_path / prompt / "input.txt"
        input_path.parent.mkdir()
        input_path.write_text(content)
        inputs[prompt] = input_path
        outputs[prompt] = tmp_path / f"{prompt}-output"

    async def run_prompt(prompt: str) -> None:
        async with agent.session(
            input_files=inputs[prompt], output_dir=outputs[prompt], cache_on_interrupt=False
        ) as session:
            finish_params, history, _ = await session.run(prompt)
        assert finish_params is not None
        assert finish_params.reason == prompt
        assert all(
            other not in str(message.content)
            for turn in history
            for message in turn
            for other in ({"first file", "second file"} - {prompt})
        )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_prompt, "first file")
        task_group.start_soon(run_prompt, "second file")

    assert (outputs["first file"] / "result.txt").read_text() == "FIRST-DATA"
    assert (outputs["second file"] / "result.txt").read_text() == "SECOND-DATA"


async def test_concurrent_sessions_with_default_loggers_can_render_while_overlapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = StringIO()
    monkeypatch.setattr(
        "stirrup.utils.logging.console",
        Console(file=output, force_terminal=False, width=120),
    )
    agent = Agent(client=ScriptedClient([]), name="concurrent-logger", tools=[])
    first_entered = anyio.Event()
    second_entered = anyio.Event()

    async def open_first_session() -> None:
        async with agent.session(cache_on_interrupt=False):
            first_entered.set()
            await second_entered.wait()

    async def open_second_session() -> None:
        await first_entered.wait()
        async with agent.session(cache_on_interrupt=False):
            second_entered.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(open_first_session)
        task_group.start_soon(open_second_session)

    assert output.getvalue().count("concurrent-logger") >= 2


async def test_cached_root_runs_with_different_task_hashes_remain_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("stirrup.core.cache.DEFAULT_CACHE_DIR", tmp_path)
    agent = Agent(
        client=ConcurrentPromptClient(),
        name="different-cached-tasks",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    reasons: set[str] = set()

    async def run_prompt(prompt: str) -> None:
        async with agent.session() as session:
            finish_params, _, _ = await session.run(prompt)
        assert finish_params is not None
        reasons.add(finish_params.reason)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_prompt, "first prompt")
        task_group.start_soon(run_prompt, "second prompt")

    assert reasons == {"first prompt", "second prompt"}


async def test_identical_root_runs_remain_concurrent_when_interrupt_caching_is_disabled() -> None:
    client = ConcurrentIdenticalClient()
    agent = Agent(
        client=client,
        name="identical-uncached-tasks",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async def run_prompt() -> None:
        async with agent.session(cache_on_interrupt=False) as session:
            finish_params, _, _ = await session.run("same prompt")
        assert finish_params is not None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_prompt)
        task_group.start_soon(run_prompt)

    assert client.call_count == 2


async def test_concurrent_identical_cached_root_run_is_rejected_before_cache_io(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("stirrup.core.cache.DEFAULT_CACHE_DIR", tmp_path)
    client = BlockingFinishClient()
    agent = Agent(
        client=client,
        name="identical-cached-tasks",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    load_count = 0
    original_load_state = CacheManager.load_state

    def count_load(cache_manager: CacheManager, task_hash: str) -> CacheState | None:
        nonlocal load_count
        load_count += 1
        return original_load_state(cache_manager, task_hash)

    monkeypatch.setattr(CacheManager, "load_state", count_load)

    async def run_first() -> None:
        async with agent.session(resume=True) as session:
            await session.run("same cached prompt")

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_first)
        await client.started.wait()

        with pytest.raises(RuntimeError, match="cannot use the same task cache"):
            async with agent.session(resume=True) as duplicate:
                await duplicate.run("same cached prompt")
        assert load_count == 1
        client.release.set()

    async with agent.session(resume=True) as reusable:
        await reusable.run("same cached prompt")

    assert load_count == 2


async def test_successful_run_keeps_task_cache_reserved_until_failed_session_teardown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("stirrup.core.cache.DEFAULT_CACHE_DIR", tmp_path)
    body_error = RuntimeError("body failed after run")
    client = ScriptedClient(
        [
            _finish_response("first", "finish-first"),
            _finish_response("after teardown", "finish-after-teardown"),
        ]
    )
    agent = Agent(
        client=client,
        name="teardown-cache-owner",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    run_finished = anyio.Event()
    fail_body = anyio.Event()
    teardown_finished = anyio.Event()

    async def finish_then_fail() -> None:
        try:
            async with agent.session(clear_cache_on_success=False) as session:
                await session.run("same cached prompt")
                run_finished.set()
                await fail_body.wait()
                raise body_error
        except RuntimeError as exc:
            assert exc is body_error
        finally:
            teardown_finished.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(finish_then_fail)
        await run_finished.wait()

        with pytest.raises(RuntimeError, match="cannot use the same task cache"):
            async with agent.session() as duplicate:
                await duplicate.run("same cached prompt")

        fail_body.set()
        await teardown_finished.wait()

    cached = CacheManager(cache_base_dir=tmp_path).load_state(compute_task_hash("same cached prompt"))
    assert cached is not None
    assert cached.agent_name == "teardown-cache-owner"

    async with agent.session() as reusable:
        finish_params, _, _ = await reusable.run("same cached prompt")

    assert finish_params is not None
    assert finish_params.reason == "after teardown"


async def test_concurrent_calls_to_same_subagent_are_isolated() -> None:
    child = Agent(
        client=ConcurrentPromptClient(),
        name="worker",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    parent = Agent(client=ScriptedClient([]), name="parent", tools=[child.to_tool()], logger=_quiet_logger())
    results: dict[str, ToolMessage] = {}

    async with parent.session(cache_on_interrupt=False) as session:

        async def call_child(prompt: str) -> None:
            results[prompt] = await session.run_tool(
                ToolCall(name="worker", arguments=json.dumps({"task": prompt}), tool_call_id=f"call-{prompt}"),
                {},
            )

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(call_child, "first prompt")
            task_group.start_soon(call_child, "second prompt")

    for prompt, other_prompt in (("first prompt", "second prompt"), ("second prompt", "first prompt")):
        assert results[prompt].success
        assert prompt in str(results[prompt].content)
        assert other_prompt not in str(results[prompt].content)


async def test_direct_parent_run_cannot_delegate_or_write_subagent_output_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    child_client = ScriptedClient(
        [
            AssistantMessage(
                content="write an output",
                tool_calls=[
                    ToolCall(
                        name="code_exec",
                        arguments='{"cmd": "printf CHILD > leaked.txt"}',
                        tool_call_id="write-child-output",
                    )
                ],
                token_usage=TokenUsage(),
            ),
            AssistantMessage(
                content="child finished",
                tool_calls=[
                    ToolCall(
                        name=DEFAULT_FINISH_TOOL_NAME,
                        arguments='{"reason": "child", "paths": ["leaked.txt"]}',
                        tool_call_id="finish-child-output",
                    )
                ],
                token_usage=TokenUsage(),
            ),
        ]
    )
    child = Agent(
        client=child_client,
        name="writer",
        max_turns=2,
        tools=[LocalCodeExecToolProvider()],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    parent = Agent(
        client=ScriptedClient(
            [
                AssistantMessage(
                    content="delegate",
                    tool_calls=[ToolCall(name="writer", arguments='{"task": "write"}', tool_call_id="delegate-writer")],
                    token_usage=TokenUsage(),
                ),
                _finish_response("parent handled error", "finish-parent"),
            ]
        ),
        name="direct-parent",
        max_turns=2,
        tools=[child.to_tool()],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    finish_params, history, _ = await parent.run("delegate without a session")

    delegated_result = next(
        message for turn in history for message in turn if isinstance(message, ToolMessage) and message.name == "writer"
    )
    assert finish_params is not None
    assert finish_params.reason == "parent handled error"
    assert not delegated_result.success
    assert "requires an active parent Agent session" in str(delegated_result.content)
    assert "async with parent.session()" in str(delegated_result.content)
    assert len(child_client.responses) == 2
    assert not (tmp_path / "leaked.txt").exists()


async def test_provider_free_base_run_releases_task_cache_reservation_after_failure() -> None:
    run_error = RuntimeError("direct run failed")
    agent = Agent(
        client=ScriptedClient([run_error, _finish_response("reused", "finish-reused")]),
        name="direct-reusable",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    with pytest.raises(RuntimeError, match="direct run failed") as exc_info:
        await agent.run("same direct prompt")
    finish_params, _, _ = await agent.run("same direct prompt")

    assert exc_info.value is run_error
    assert finish_params is not None
    assert finish_params.reason == "reused"


async def test_provider_free_base_run_is_independent_inside_an_active_session(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "outer_skill")
    input_path = tmp_path / "outer_input.txt"
    input_path.write_text("outer data")

    inner_failure = RuntimeError("inner run failed")
    inner_client = ScriptedClient(
        [
            AssistantMessage(
                content="finish independently",
                tool_calls=[
                    ToolCall(
                        name=DEFAULT_FINISH_TOOL_NAME,
                        arguments='{"reason": "inner", "paths": ["missing-from-outer.txt"]}',
                        tool_call_id="finish-inner",
                    )
                ],
                token_usage=TokenUsage(),
            ),
            inner_failure,
        ]
    )
    inner_logger = WarningRecordingLogger()
    inner = Agent(
        client=inner_client,
        name="inner",
        max_turns=1,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=inner_logger,
    )
    outer = Agent(
        client=ScriptedClient([]),
        name="outer",
        tools=[LocalCodeExecToolProvider()],
        logger=_quiet_logger(),
    )

    async with outer.session(
        input_files=input_path,
        skills_dir=skills_dir,
        cache_on_interrupt=False,
    ) as outer_session:
        finish_params, _, _ = await inner.run("independent work")
        with pytest.raises(RuntimeError, match="inner run failed") as exc_info:
            await inner.run("failing independent work")
        outer_result = await outer_session.run_tool(
            ToolCall(name="code_exec", arguments='{"cmd": "printf OUTER"}', tool_call_id="outer-restored"),
            {},
        )

    assert finish_params is not None
    assert finish_params.reason == "inner"
    assert exc_info.value is inner_failure
    assert outer_result.success
    assert "OUTER" in str(outer_result.content)
    inner_system_prompt = next(
        message for message in inner_client.messages_seen[0] if isinstance(message, SystemMessage)
    )
    assert "outer_input.txt" not in str(inner_system_prompt.content)
    assert "outer_skill" not in str(inner_system_prompt.content)
    assert all("no output_dir is set" not in warning for warning in inner_logger.warnings_seen)


async def test_nested_root_is_independent_while_true_subagent_inherits_skills(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "shared_skill")
    child_client = ScriptedClient([_finish_response("child", "finish-child")])
    root_client = ScriptedClient([_finish_response("root", "finish-root")])
    child = Agent(
        client=child_client,
        name="skill_child",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    parent = Agent(client=ScriptedClient([]), name="parent", tools=[child.to_tool()], logger=_quiet_logger())
    independent_root = Agent(
        client=root_client,
        name="independent",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async with parent.session(skills_dir=skills_dir, cache_on_interrupt=False) as parent_session:
        async with independent_root.session(cache_on_interrupt=False) as root_session:
            await root_session.run("independent work")
        child_result = await parent_session.run_tool(
            ToolCall(name="skill_child", arguments='{"task": "delegated work"}', tool_call_id="delegate"),
            {},
        )

    assert child_result.success
    child_system = [message for message in child_client.messages_seen[0] if isinstance(message, SystemMessage)]
    root_system = [message for message in root_client.messages_seen[0] if isinstance(message, SystemMessage)]
    assert any("shared_skill" in str(message.content) for message in child_system)
    assert all("shared_skill" not in str(message.content) for message in root_system)


async def test_session_run_and_run_tool_require_their_exact_active_ambient_context() -> None:
    client = ScriptedClient([])
    tool_execution_count = 0

    def count_execution(_params: BaseModel) -> ToolResult:
        nonlocal tool_execution_count
        tool_execution_count += 1
        return ToolResult(content="executed")

    agent = Agent(
        client=client,
        name="guarded",
        tools=[Tool(name="count_execution", description="Count executions", executor=count_execution)],
        logger=_quiet_logger(),
    )
    first = agent.session(cache_on_interrupt=False)
    tool_call = ToolCall(name="count_execution", arguments="{}", tool_call_id="count")

    with pytest.raises(RuntimeError, match="own active session context"):
        await first.run("before")
    with pytest.raises(RuntimeError, match="own active session context"):
        await first.run_tool(tool_call, {})

    async with first:
        valid_result = await first.run_tool(tool_call, {})
        async with agent.session(cache_on_interrupt=False):
            with pytest.raises(RuntimeError, match="own active session context"):
                await first.run("wrong ambient")
            with pytest.raises(RuntimeError, match="own active session context"):
                await first.run_tool(tool_call, {})
        restored_result = await first.run_tool(tool_call, {})

    with pytest.raises(RuntimeError, match="own active session context"):
        await first.run("after")
    with pytest.raises(RuntimeError, match="own active session context"):
        await first.run_tool(tool_call, {})

    assert valid_result.success
    assert restored_result.success
    assert tool_execution_count == 2
    assert client.messages_seen == []


async def test_failed_session_entry_restores_outer_context_and_releases_custom_resources() -> None:
    provider = CustomProvider()
    failing = Agent(
        client=ScriptedClient([]),
        name="failing",
        tools=[provider, ViewImageToolProvider()],
        logger=_quiet_logger(),
    )
    outer = Agent(
        client=ScriptedClient([_finish_response("outer", "finish-outer")]),
        name="outer",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async with outer.session(cache_on_interrupt=False) as outer_session:
        with pytest.raises(RuntimeError, match="requires a CodeExecToolProvider"):
            async with failing.session(cache_on_interrupt=False):
                pass
        finish_params, _, _ = await outer_session.run("outer still active")

    assert finish_params is not None
    assert finish_params.reason == "outer"
    assert (provider.enter_count, provider.exit_count) == (1, 1)
    reusable = Agent(client=ScriptedClient([]), name="reusable", tools=[provider], logger=_quiet_logger())
    async with reusable.session(cache_on_interrupt=False):
        pass


async def test_custom_provider_and_logger_remain_reusable_sequentially() -> None:
    provider = CustomProvider()
    custom_logger = CustomLogger()
    agent = Agent(client=ScriptedClient([]), name="custom", tools=[provider], logger=custom_logger)

    for _ in range(2):
        async with agent.session(cache_on_interrupt=False) as session:
            result = await session.run_tool(
                ToolCall(name="inspect_custom", arguments="{}", tool_call_id="inspect"),
                {},
            )
            assert result.success

    assert (provider.enter_count, provider.exit_count) == (2, 2)
    assert (custom_logger.enter_count, custom_logger.exit_count) == (2, 2)


async def test_cancellation_shields_cleanup_and_holds_custom_reservation_until_it_finishes() -> None:
    provider = BlockingCleanupProvider()
    agent = Agent(client=ScriptedClient([]), name="cancel-cleanup", tools=[provider], logger=_quiet_logger())
    entered = anyio.Event()
    cancelled_session_finished = anyio.Event()
    cancel_scope: anyio.CancelScope | None = None

    async def cancel_during_session() -> None:
        nonlocal cancel_scope
        with anyio.CancelScope() as scope:
            cancel_scope = scope
            async with agent.session(cache_on_interrupt=False):
                entered.set()
                await anyio.sleep_forever()
        cancelled_session_finished.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(cancel_during_session)
        await entered.wait()
        assert cancel_scope is not None
        cancel_scope.cancel()
        await provider.cleanup_started.wait()

        overlapping = agent.session(cache_on_interrupt=False)
        try:
            with pytest.raises(RuntimeError, match="sequential sessions only"):
                await overlapping.__aenter__()
        finally:
            provider.cleanup_allowed.set()
            if overlapping._session_state is not None:  # noqa: SLF001
                await overlapping.__aexit__(None, None, None)

        await provider.cleanup_finished.wait()
        await cancelled_session_finished.wait()

    async with agent.session(cache_on_interrupt=False):
        pass

    assert (provider.enter_count, provider.exit_count) == (2, 2)
    assert isinstance(provider.first_exit_exception, anyio.get_cancelled_exc_class())
    assert any(
        "Tool provider failed during session cleanup" in note for note in provider.first_exit_exception.__notes__
    )


async def test_shared_custom_provider_overlap_across_agents_is_rejected_before_entry() -> None:
    provider = CustomProvider()
    first = Agent(client=ScriptedClient([]), name="first", tools=[provider], logger=_quiet_logger())
    second = Agent(client=ScriptedClient([]), name="second", tools=[provider], logger=_quiet_logger())

    async with first.session(cache_on_interrupt=False):
        with pytest.raises(RuntimeError, match="sequential sessions only"):
            async with second.session(cache_on_interrupt=False):
                pass
        assert provider.enter_count == 1

    assert provider.exit_count == 1


async def test_shared_custom_logger_overlap_across_agents_is_rejected_before_entry() -> None:
    custom_logger = CustomLogger()
    first = Agent(client=ScriptedClient([]), name="first", tools=[], logger=custom_logger)
    second = Agent(client=ScriptedClient([]), name="second", tools=[], logger=custom_logger)

    async with first.session(cache_on_interrupt=False):
        with pytest.raises(RuntimeError, match="sequential sessions only"):
            async with second.session(cache_on_interrupt=False):
                pass
        assert custom_logger.enter_count == 1

    assert custom_logger.exit_count == 1


async def test_view_image_uses_each_sessions_active_code_backend() -> None:
    configured_exec = LocalCodeExecToolProvider()
    agent = Agent(
        client=ScriptedClient([]),
        name="images",
        tools=[configured_exec, ViewImageToolProvider(configured_exec)],
        logger=_quiet_logger(),
    )
    arrived = {"red": anyio.Event(), "blue": anyio.Event()}
    session_execs: dict[str, LocalCodeExecToolProvider] = {}

    async def view_in_session(color: str, other_color: str) -> None:
        async with agent.session(cache_on_interrupt=False) as session:
            session_tools = session._tools  # noqa: SLF001
            session_exec = next(tool for tool in session_tools if isinstance(tool, LocalCodeExecToolProvider))
            session_execs[color] = session_exec
            arrived[color].set()
            await arrived[other_color].wait()

            image_path = session_exec.temp_dir / "pixel.png"  # type: ignore[operator]
            Image.new("RGB", (1, 1), color=color).save(image_path)
            result = await session.run_tool(
                ToolCall(name="view_image", arguments='{"path": "pixel.png"}', tool_call_id=f"view-{color}"),
                {},
            )
            assert result.success

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(view_in_session, "red", "blue")
        task_group.start_soon(view_in_session, "blue", "red")

    assert session_execs["red"] is not session_execs["blue"]


async def test_custom_view_image_provider_and_exact_backend_are_reusable_sequentially() -> None:
    configured_exec = LocalCodeExecToolProvider()
    view_images = CountingViewImageProvider(configured_exec)
    agent = Agent(
        client=ScriptedClient([]),
        name="custom-images",
        tools=[configured_exec, view_images],
        logger=_quiet_logger(),
    )

    for index, color in enumerate(("red", "blue")):
        async with agent.session(cache_on_interrupt=False) as session:
            session_tools = session._tools  # noqa: SLF001
            session_exec = next(tool for tool in session_tools if isinstance(tool, LocalCodeExecToolProvider))
            assert session_exec is configured_exec
            image_path = configured_exec.temp_dir / "pixel.png"  # type: ignore[operator]
            Image.new("RGB", (1, 1), color=color).save(image_path)
            result = await session.run_tool(
                ToolCall(name="view_image", arguments='{"path": "pixel.png"}', tool_call_id=f"view-{index}"),
                {},
            )
            assert result.success

    assert view_images.enter_count == 2
    assert configured_exec.temp_dir is None


async def test_custom_view_image_pair_rejects_shared_backend_overlap_before_entry() -> None:
    shared_exec = LocalCodeExecToolProvider()
    first_view_images = CountingViewImageProvider(shared_exec)
    second_view_images = CountingViewImageProvider(shared_exec)
    first = Agent(
        client=ScriptedClient([]),
        name="first-images",
        tools=[shared_exec, first_view_images],
        logger=_quiet_logger(),
    )
    second = Agent(
        client=ScriptedClient([]),
        name="second-images",
        tools=[shared_exec, second_view_images],
        logger=_quiet_logger(),
    )

    async with first.session(cache_on_interrupt=False):
        with pytest.raises(RuntimeError, match="sequential sessions only"):
            async with second.session(cache_on_interrupt=False):
                pass
        assert first_view_images.enter_count == 1
        assert second_view_images.enter_count == 0


async def test_shared_subagent_rejects_custom_view_image_backend_pair_before_child_entry() -> None:
    child_exec = LocalCodeExecToolProvider()
    view_images = CountingViewImageProvider(child_exec)
    child = Agent(
        client=ScriptedClient([]),
        name="custom_shared_child",
        tools=[child_exec, view_images],
        share_parent_exec_env=True,
        logger=_quiet_logger(),
    )
    parent = Agent(
        client=ScriptedClient([]),
        name="custom_shared_parent",
        tools=[LocalCodeExecToolProvider(), child.to_tool()],
        logger=_quiet_logger(),
    )

    async with parent.session(cache_on_interrupt=False) as session:
        child_result = await session.run_tool(
            ToolCall(name="custom_shared_child", arguments='{"task": "view it"}', tool_call_id="delegate-custom"),
            {},
        )
        parent_result = await session.run_tool(
            ToolCall(name="code_exec", arguments='{"cmd": "printf PARENT"}', tool_call_id="parent-after-reject"),
            {},
        )

    assert not child_result.success
    assert "cannot use share_parent_exec_env=True" in str(child_result.content)
    assert view_images.enter_count == 0
    assert child_exec.temp_dir is None
    assert parent_result.success


async def test_shared_subagent_view_image_uses_parent_code_backend(tmp_path: Path) -> None:
    pixel_path = tmp_path / "pixel.png"
    Image.new("RGB", (1, 1), color="red").save(pixel_path)
    child_exec = LocalCodeExecToolProvider()
    child = Agent(
        client=ScriptedClient(
            [
                AssistantMessage(
                    content="view shared image",
                    tool_calls=[
                        ToolCall(name="view_image", arguments='{"path": "pixel.png"}', tool_call_id="view-shared")
                    ],
                    token_usage=TokenUsage(),
                ),
                _finish_response("shared", "finish-shared"),
            ]
        ),
        name="shared_child",
        max_turns=2,
        tools=[child_exec, ViewImageToolProvider(child_exec)],
        finish_tool=SIMPLE_FINISH_TOOL,
        share_parent_exec_env=True,
        logger=_quiet_logger(),
    )
    parent = Agent(
        client=ScriptedClient([]),
        name="shared_parent",
        tools=[LocalCodeExecToolProvider(), child.to_tool()],
        logger=_quiet_logger(),
    )
    run_metadata: dict[str, list[Any]] = {}

    async with parent.session(input_files=pixel_path, cache_on_interrupt=False) as session:
        child_result = await session.run_tool(
            ToolCall(name="shared_child", arguments='{"task": "view it"}', tool_call_id="delegate-shared"),
            run_metadata,
        )
        parent_result = await session.run_tool(
            ToolCall(name="code_exec", arguments='{"cmd": "printf PARENT"}', tool_call_id="parent-still-active"),
            {},
        )

    assert child_result.success
    assert parent_result.success
    assert "PARENT" in str(parent_result.content)
    child_metadata = run_metadata["shared_child"][0]
    assert isinstance(child_metadata, SubAgentMetadata)
    view_results = [
        message
        for turn in child_metadata.message_history
        for message in turn
        if isinstance(message, ToolMessage) and message.name == "view_image"
    ]
    assert len(view_results) == 1
    assert view_results[0].success


async def test_partial_entry_cleanup_failures_do_not_mask_the_entry_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CleanupFailingProvider()
    cleanup_logger = CleanupFailingLogger()
    entry_error = RuntimeError("session entry failed")

    def fail_interrupt_installation() -> bool:
        raise entry_error

    monkeypatch.setattr("stirrup.core.agent._install_interrupt_handler", fail_interrupt_installation)
    agent = Agent(
        client=ScriptedClient([]),
        name="entry-cleanup",
        tools=[provider],
        logger=cleanup_logger,
    )

    with pytest.raises(RuntimeError, match="session entry failed") as exc_info:
        async with agent.session():
            pass

    assert exc_info.value is entry_error
    assert (provider.enter_count, provider.exit_count) == (1, 1)
    assert (cleanup_logger.enter_count, cleanup_logger.exit_count) == (1, 1)
    assert any("Logger failed during session cleanup" in note for note in entry_error.__notes__)
    assert any("Tool provider failed during session cleanup" in note for note in entry_error.__notes__)


async def test_body_failure_survives_all_cleanup_failures() -> None:
    provider = CleanupFailingProvider()
    cleanup_logger = CleanupFailingLogger()
    body_error = RuntimeError("session body failed")
    agent = Agent(
        client=ScriptedClient([]),
        name="body-cleanup",
        tools=[provider],
        logger=cleanup_logger,
    )

    with pytest.raises(RuntimeError, match="session body failed") as exc_info:
        async with agent.session(cache_on_interrupt=False):
            raise body_error

    assert exc_info.value is body_error
    assert (provider.enter_count, provider.exit_count) == (1, 1)
    assert (cleanup_logger.enter_count, cleanup_logger.exit_count) == (1, 1)
    assert any("Logger failed during session cleanup" in note for note in body_error.__notes__)
    assert any("Tool provider failed during session cleanup" in note for note in body_error.__notes__)


async def test_cleanup_failure_is_raised_when_the_session_has_no_primary_failure() -> None:
    provider = CleanupFailingProvider()
    cleanup_logger = CleanupFailingLogger()
    agent = Agent(
        client=ScriptedClient([]),
        name="cleanup-only",
        tools=[provider],
        logger=cleanup_logger,
    )

    with pytest.raises(RuntimeError, match="logger cleanup failed") as exc_info:
        async with agent.session(cache_on_interrupt=False):
            pass

    assert exc_info.value is cleanup_logger.exit_error
    assert provider.exit_count == 1
    assert any("Tool provider failed during session cleanup" in note for note in exc_info.value.__notes__)


async def test_worker_thread_failure_is_cached_without_masking_original_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("stirrup.core.cache.DEFAULT_CACHE_DIR", tmp_path)
    original_error = RuntimeError("client failed")
    prompt = "cache this worker failure"

    def run_in_worker() -> None:
        async def fail_run() -> None:
            agent = Agent(
                client=ScriptedClient([original_error]),
                name="worker-cache",
                tools=[],
                logger=_quiet_logger(),
            )
            async with agent.session() as session:
                await session.run(prompt)

        anyio.run(fail_run)

    with pytest.raises(RuntimeError, match="client failed") as exc_info:
        await anyio.to_thread.run_sync(run_in_worker)  # ty: ignore[unresolved-attribute]

    assert exc_info.value is original_error
    cached = CacheManager(cache_base_dir=tmp_path).load_state(compute_task_hash(prompt))
    assert cached is not None
    assert cached.agent_name == "worker-cache"


async def test_overlapping_root_sessions_restore_signal_handler_after_last_exit() -> None:
    agent = Agent(client=ScriptedClient([]), name="signals", tools=[], logger=_quiet_logger())
    first_entered = anyio.Event()
    second_entered = anyio.Event()
    first_exited = anyio.Event()
    original_handler = signal.getsignal(signal.SIGINT)
    installed_handler: object | None = None

    async def first_session() -> None:
        nonlocal installed_handler
        async with agent.session():
            installed_handler = signal.getsignal(signal.SIGINT)
            first_entered.set()
            await second_entered.wait()
        first_exited.set()

    async def second_session() -> None:
        await first_entered.wait()
        async with agent.session():
            second_entered.set()
            await first_exited.wait()
            assert signal.getsignal(signal.SIGINT) is installed_handler

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(first_session)
        task_group.start_soon(second_session)

    assert signal.getsignal(signal.SIGINT) is original_handler
