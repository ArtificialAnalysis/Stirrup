"""Behavioral contracts for resumable accepted agent work."""

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pydantic import BaseModel

from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core import cache as cache_module
from stirrup.core.agent import Agent
from stirrup.core.cache import CacheManager
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    ImageContentBlock,
    LLMClient,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from stirrup.tools.code_backends.base import (
    CodeExecToolProvider,
    SaveOutputFilesResult,
)
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL, FinishParams
from stirrup.utils.logging import AgentLogger


class ScriptedClient(LLMClient):
    def __init__(self, responses: list[AssistantMessage | Exception]) -> None:
        self.responses = list(responses)
        self.messages_seen: list[list[ChatMessage]] = []

    @property
    def model_slug(self) -> str:
        return "resume-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        del tools
        self.messages_seen.append(list(messages))
        if not self.responses:
            raise AssertionError("unexpected model generation")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class LabelParams(BaseModel):
    label: str


class ReceiptMetadata(BaseModel):
    labels: list[str]

    def __add__(self, other: "ReceiptMetadata") -> "ReceiptMetadata":
        return type(self)(labels=self.labels + other.labels)


class FailFirstToolLogger(AgentLogger):
    def __init__(self) -> None:
        super().__init__(show_spinner=False)
        self.has_failed = False

    def tool_result(self, tool_message: ToolMessage) -> None:
        if not self.has_failed:
            self.has_failed = True
            raise RuntimeError("tool logger failed")
        super().tool_result(tool_message)


class FailImageUserLogger(AgentLogger):
    def __init__(self) -> None:
        super().__init__(show_spinner=False)
        self.has_failed = False

    def user_message(self, user_message: UserMessage) -> None:
        has_image = isinstance(user_message.content, list) and any(
            isinstance(block, ImageContentBlock) for block in user_message.content
        )
        if has_image and not self.has_failed:
            self.has_failed = True
            raise RuntimeError("image logger failed")
        super().user_message(user_message)


class FailStepLogger(AgentLogger):
    def __init__(self) -> None:
        super().__init__(show_spinner=False)
        self.has_failed = False

    def on_step(
        self,
        step: int,
        tool_calls: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        del step, tool_calls, input_tokens, output_tokens
        if not self.has_failed:
            self.has_failed = True
            raise RuntimeError("step callback failed")


class FailExitLogger(AgentLogger):
    def __init__(self) -> None:
        super().__init__(show_spinner=False)
        self.has_failed = False

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        super().__exit__(exc_type, exc_val, exc_tb)
        if not self.has_failed:
            self.has_failed = True
            raise RuntimeError("logger teardown failed")


class RemoteOnlyProvider(LocalCodeExecToolProvider):
    @property
    def temp_dir(self) -> None:
        return None


class ExportFailOnceProvider(LocalCodeExecToolProvider):
    def __init__(self) -> None:
        super().__init__()
        self.fail_export = True

    async def save_output_files(
        self,
        paths: list[str],
        output_dir: str | Path,
        dest_env: CodeExecToolProvider | None = None,
    ) -> SaveOutputFilesResult:
        del paths, output_dir, dest_env
        if self.fail_export:
            self.fail_export = False
            return SaveOutputFilesResult(failed={"report.txt": "export failed"})
        return SaveOutputFilesResult()


class TeardownFailOnceProvider(ExportFailOnceProvider):
    def __init__(self) -> None:
        super().__init__()
        self.fail_export = False
        self.fail_teardown = True

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await super().__aexit__(exc_type, exc_val, exc_tb)
        if self.fail_teardown:
            self.fail_teardown = False
            raise RuntimeError("provider teardown failed")


def _quiet_logger() -> AgentLogger:
    return AgentLogger(show_spinner=False)


def _finish(reason: str, call_id: str = "finish", paths: list[str] | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=reason,
        tool_calls=[
            ToolCall(
                name=DEFAULT_FINISH_TOOL_NAME,
                arguments=json.dumps({"reason": reason, "paths": paths or []}),
                tool_call_id=call_id,
            )
        ],
        token_usage=TokenUsage(input=5, answer=2),
    )


def _recording_tool(executions: list[str]) -> Tool[LabelParams, ReceiptMetadata]:
    def execute(params: LabelParams) -> ToolResult[ReceiptMetadata]:
        executions.append(params.label)
        return ToolResult(content=params.label, metadata=ReceiptMetadata(labels=[params.label]))

    return Tool(
        name="record",
        description="record once",
        parameters=LabelParams,
        executor=execute,
    )


def _cache_state_json(cache_root: Path) -> dict[str, Any]:
    task_hash = CacheManager(cache_base_dir=cache_root).list_caches()[0]
    pointer = json.loads((cache_root / task_hash / "current.json").read_text())
    state_path = cache_root / task_hash / "generations" / pointer["generation"] / "state.json"
    return json.loads(state_path.read_text())


async def test_partial_batch_resumes_only_pending_calls_with_flat_typed_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(cache_module, "DEFAULT_CACHE_DIR", cache_root)
    executions: list[str] = []
    record = _recording_tool(executions)
    ordered_turn = AssistantMessage(
        content="record twice",
        tool_calls=[
            ToolCall(name="record", arguments='{"label": "first"}', tool_call_id="first"),
            ToolCall(name="record", arguments='{"label": "second"}', tool_call_id="second"),
        ],
        token_usage=TokenUsage(input=10, answer=4),
    )
    first = Agent(
        client=ScriptedClient([ordered_turn]),
        name="partial-agent",
        max_turns=2,
        tools=[record],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=FailFirstToolLogger(),
    )

    with pytest.raises(RuntimeError, match="tool logger failed"):
        async with first.session(resume=True, clear_cache_on_success=False) as session:
            await session.run("record durably")

    serialized = _cache_state_json(cache_root)
    assert set(serialized).isdisjoint(
        {"finish_tool_name", "finish_params", "pending_assistant_id", "pending_tool_calls", "run_metadata_by_turn"}
    )
    assert list(serialized["run_metadata"]) == ["record"]

    resume_client = ScriptedClient([_finish("resumed")])
    resumed = Agent(
        client=resume_client,
        name="partial-agent",
        max_turns=2,
        tools=[record],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    async with resumed.session(resume=True, clear_cache_on_success=False) as session:
        finish_params, history, metadata = await session.run("record durably")

    assert finish_params is not None and finish_params.reason == "resumed"
    assert executions == ["first", "second"]
    assert resume_client.messages_seen and [
        message.tool_call_id for message in resume_client.messages_seen[0] if isinstance(message, ToolMessage)
    ] == ["first", "second"]
    assert metadata["record"] == [ReceiptMetadata(labels=["first"]), ReceiptMetadata(labels=["second"])]
    restored = [message for group in history for message in group if isinstance(message, ToolMessage)]
    assert [message.tool_call_id for message in restored] == ["first", "second", "finish"]


async def test_atomic_text_only_image_result_resumes_later_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(cache_module, "DEFAULT_CACHE_DIR", cache_root)
    image_path = tmp_path / "pixel.png"
    Image.new("RGB", (1, 1), color="red").save(image_path)
    image_block = ImageContentBlock(data=image_path.read_bytes())
    image_executions: list[str] = []
    record_executions: list[str] = []

    def capture(params: LabelParams) -> ToolResult:
        image_executions.append(params.label)
        return ToolResult(content=["captured", image_block])

    image_tool = Tool(name="image", description="capture", parameters=LabelParams, executor=capture)
    record_tool = _recording_tool(record_executions)
    turn = AssistantMessage(
        content="capture then record",
        tool_calls=[
            ToolCall(name="image", arguments='{"label": "image"}', tool_call_id="image"),
            ToolCall(name="record", arguments='{"label": "record"}', tool_call_id="record"),
        ],
        token_usage=TokenUsage(input=10, answer=4),
    )
    first = Agent(
        client=ScriptedClient([turn]),
        name="image-agent",
        max_turns=2,
        tools=[image_tool, record_tool],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=FailImageUserLogger(),
    )
    with pytest.raises(RuntimeError, match="image logger failed"):
        async with first.session(resume=True) as session:
            await session.run("capture once")

    resumed_client = ScriptedClient([_finish("image resumed")])
    resumed = Agent(
        client=resumed_client,
        name="image-agent",
        max_turns=2,
        tools=[image_tool, record_tool],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    async with resumed.session(resume=True) as session:
        finish_params, _, _ = await session.run("capture once")

    assert finish_params is not None
    assert image_executions == ["image"]
    assert record_executions == ["record"]
    request = resumed_client.messages_seen[0]
    assert [message.tool_call_id for message in request if isinstance(message, ToolMessage)] == ["image", "record"]
    assert (
        sum(
            isinstance(message, UserMessage)
            and isinstance(message.content, list)
            and any(isinstance(block, ImageContentBlock) for block in message.content)
            for message in request
        )
        == 1
    )


@pytest.mark.parametrize("failure_kind", ["callback", "logger-exit", "provider-exit", "output-export"])
async def test_terminal_checkpoint_survives_post_finish_failures_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    cache_root = tmp_path / failure_kind
    monkeypatch.setattr(cache_module, "DEFAULT_CACHE_DIR", cache_root)
    finish_executions: list[str] = []

    def finish_once(params: FinishParams) -> ToolResult[ReceiptMetadata]:
        finish_executions.append(params.reason)
        return ToolResult(content=params.reason, metadata=ReceiptMetadata(labels=[params.reason]))

    finish_tool = Tool(
        name=DEFAULT_FINISH_TOOL_NAME,
        description="finish once",
        parameters=FinishParams,
        executor=finish_once,
    )
    logger: AgentLogger = _quiet_logger()
    tools: list[Tool | CodeExecToolProvider] = []
    output_dir: Path | None = None
    provider: CodeExecToolProvider | None = None
    expected_error = ""
    if failure_kind == "callback":
        logger = FailStepLogger()
        expected_error = "step callback failed"
    elif failure_kind == "logger-exit":
        logger = FailExitLogger()
        expected_error = "logger teardown failed"
    elif failure_kind == "provider-exit":
        provider = TeardownFailOnceProvider()
        tools = [provider]
        expected_error = "provider teardown failed"
    else:
        provider = ExportFailOnceProvider()
        tools = [provider]
        output_dir = tmp_path / "output"
        expected_error = "Failed to export output files"

    first = Agent(  # ty: ignore[no-matching-overload]
        client=ScriptedClient([_finish("terminal", paths=["report.txt"] if output_dir else [])]),
        name=f"terminal-{failure_kind}",
        max_turns=1,
        tools=tools,
        finish_tool=finish_tool,
        logger=logger,
    )
    with pytest.raises(RuntimeError, match=expected_error):
        async with first.session(resume=True, output_dir=output_dir) as session:
            await session.run("finish once")

    resumed = Agent(
        client=ScriptedClient([]),
        name=f"terminal-{failure_kind}",
        max_turns=1,
        tools=[provider] if provider is not None else [],
        finish_tool=finish_tool,
        logger=_quiet_logger(),
    )
    async with resumed.session(resume=True, output_dir=output_dir) as session:
        finish_params, history, metadata = await session.run("finish once")

    assert finish_params is not None and finish_params.reason == "terminal"
    assert finish_executions == ["terminal"]
    assert metadata[DEFAULT_FINISH_TOOL_NAME] == [ReceiptMetadata(labels=["terminal"])]
    assert (
        sum(
            isinstance(message, ToolMessage) and message.name == DEFAULT_FINISH_TOOL_NAME
            for group in history
            for message in group
        )
        == 1
    )
    assert CacheManager(cache_base_dir=cache_root).list_caches() == []


async def test_exact_restore_preserves_cached_deletion_after_identity_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(cache_module, "DEFAULT_CACHE_DIR", cache_root)
    source = tmp_path / "input.txt"
    source.write_text("identity bytes")
    delete_turn = AssistantMessage(
        content="delete input",
        tool_calls=[ToolCall(name="code_exec", arguments='{"cmd": "rm input.txt"}', tool_call_id="delete")],
        token_usage=TokenUsage(input=10, answer=2),
    )
    first = Agent(
        client=ScriptedClient([delete_turn]),
        name="deletion-agent",
        max_turns=3,
        tools=[LocalCodeExecToolProvider()],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=FailFirstToolLogger(),
    )
    with pytest.raises(RuntimeError, match="tool logger failed"):
        async with first.session(input_files=source, resume=True) as session:
            await session.run("delete the input")

    verify_turn = AssistantMessage(
        content="verify deletion",
        tool_calls=[
            ToolCall(
                name="code_exec",
                arguments='{"cmd": "test ! -e input.txt && printf deleted"}',
                tool_call_id="verify",
            )
        ],
        token_usage=TokenUsage(input=10, answer=2),
    )
    resumed_client = ScriptedClient([verify_turn, _finish("deleted")])
    resumed = Agent(
        client=resumed_client,
        name="deletion-agent",
        max_turns=3,
        tools=[LocalCodeExecToolProvider()],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    async with resumed.session(input_files=source, resume=True) as session:
        finish_params, history, _ = await session.run("delete the input")

    assert finish_params is not None and finish_params.reason == "deleted"
    delete_results = [
        message
        for group in history
        for message in group
        if isinstance(message, ToolMessage) and message.tool_call_id == "delete"
    ]
    verify_result = next(
        message
        for group in history
        for message in group
        if isinstance(message, ToolMessage) and message.tool_call_id == "verify"
    )
    assert len(delete_results) == 1
    assert verify_result.success
    assert "deleted" in str(verify_result.content)


async def test_remote_only_provider_starts_fresh_instead_of_partially_resuming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_root = tmp_path / "remote-cache"
    monkeypatch.setattr(cache_module, "DEFAULT_CACHE_DIR", cache_root)
    first_turn = AssistantMessage(
        content="cached remote turn",
        tool_calls=[],
        token_usage=TokenUsage(input=10, answer=2),
    )
    provider = RemoteOnlyProvider()
    first = Agent(
        client=ScriptedClient([first_turn]),
        name="remote-agent",
        max_turns=1,
        tools=[provider],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    async with first.session(resume=True, clear_cache_on_success=False) as session:
        finish_params, _, _ = await session.run("remote task")
    assert finish_params is None
    assert CacheManager(cache_base_dir=cache_root).list_caches()

    resumed_client = ScriptedClient([_finish("fresh remote")])
    resumed = Agent(
        client=resumed_client,
        name="remote-agent",
        max_turns=1,
        tools=[provider],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    async with resumed.session(resume=True) as session:
        finish_params, _, _ = await session.run("remote task")

    assert finish_params is not None and finish_params.reason == "fresh remote"
    assert all(
        not isinstance(message, AssistantMessage) or message.content != "cached remote turn"
        for message in resumed_client.messages_seen[0]
    )


async def test_context_overflow_summarizes_older_history_without_replaying_accepted_metadata() -> None:
    executions: list[str] = []
    record = _recording_tool(executions)
    first_turn = AssistantMessage(content="first", tool_calls=[], token_usage=TokenUsage(input=10, answer=2))
    durable_turn = AssistantMessage(
        content="durable",
        tool_calls=[ToolCall(name="record", arguments='{"label": "once"}', tool_call_id="once")],
        token_usage=TokenUsage(input=10, answer=2),
    )
    client = ScriptedClient(
        [
            first_turn,
            durable_turn,
            ContextOverflowError("overflow"),
            AssistantMessage(content="summary", tool_calls=[], token_usage=TokenUsage()),
            _finish("recovered"),
        ]
    )
    agent = Agent(
        client=client,
        name="context-agent",
        max_turns=3,
        tools=[record],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async with agent.session(cache_on_interrupt=False) as session:
        finish_params, history, metadata = await session.run("recover context")

    assert finish_params is not None and finish_params.reason == "recovered"
    assert executions == ["once"]
    assert metadata["record"] == [ReceiptMetadata(labels=["once"])]
    assert (
        sum(
            isinstance(message, ToolMessage) and message.tool_call_id == "once"
            for group in history
            for message in group
        )
        == 1
    )
