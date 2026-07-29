"""Behavioural contracts for identity-matched cache resumption.

A cached run resumes only when every model-visible input still matches. These tests drive
that through `session()`/`run()` and assert on what a user can observe: whether the resumed
run's first LLM call was shown an earlier transcript, whether already-accepted tool calls ran
again, and which of the three cache outcomes was reported.
"""

import json
import shutil
from pathlib import Path

import pytest
from pydantic import BaseModel

from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core.agent import Agent
from stirrup.core.cache import compute_task_hash
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    LLMClient,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    ToolResult,
    UserMessage,
)
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL
from stirrup.utils.logging import AgentLogger

PROMPT = "summarize the input"


class RecordingClient(LLMClient):
    """Replay scripted turns and record the messages each call was shown."""

    def __init__(self, responses: list[AssistantMessage | Exception], *, model_slug: str = "identity-model") -> None:
        self.responses = responses
        self.calls: list[list[ChatMessage]] = []
        self._model_slug = model_slug

    @property
    def model_slug(self) -> str:
        return self._model_slug

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        del tools
        self.calls.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingLogger(AgentLogger):
    """Capture the user-facing cache outcome lines instead of printing them."""

    def __init__(self) -> None:
        super().__init__(show_spinner=False)
        self.messages: list[str] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)

    def warning(self, message: str, *args: object) -> None:
        self.messages.append(message % args if args else message)


class NoteParams(BaseModel):
    note: str


def _note_tool(recorded: list[str], *, description: str = "Record a note") -> Tool[NoteParams, None]:
    def record(params: NoteParams) -> ToolResult[None]:
        recorded.append(params.note)
        return ToolResult(content=f"recorded {params.note}")

    return Tool[NoteParams, None](
        name="note",
        description=description,
        parameters=NoteParams,
        executor=record,
    )


def _note_turn(note: str, tool_call_id: str) -> AssistantMessage:
    return AssistantMessage(
        content=f"noting {note}",
        tool_calls=[ToolCall(name="note", arguments=json.dumps({"note": note}), tool_call_id=tool_call_id)],
        token_usage=TokenUsage(),
    )


def _finish_turn(reason: str, tool_call_id: str) -> AssistantMessage:
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


def _agent(
    client: LLMClient,
    tools: list[Tool | LocalCodeExecToolProvider],
    *,
    system_prompt: str | None = None,
    max_turns: int = 10,
    logger: AgentLogger | None = None,
) -> Agent:
    return Agent(
        client=client,
        name="identity",
        max_turns=max_turns,
        system_prompt=system_prompt,
        tools=list(tools),
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=logger if logger is not None else AgentLogger(show_spinner=False),
    )


def _was_resumed(client: RecordingClient) -> bool:
    """Whether the run's first LLM call was shown a transcript from an earlier run."""
    return any(isinstance(message, AssistantMessage) for message in client.calls[0])


def _kept_note_result(client: RecordingClient, note: str) -> bool:
    """Whether the run's first LLM call was shown a tool result accepted by an earlier run."""
    return any(
        isinstance(message, ToolMessage) and f"recorded {note}" in str(message.content) for message in client.calls[0]
    )


def _cache_dir(cache_base_dir: Path) -> Path:
    """The single cache directory a run left behind."""
    directories = sorted(path for path in cache_base_dir.iterdir() if path.is_dir())
    assert len(directories) == 1
    return directories[0]


@pytest.fixture(autouse=True)
def cache_base_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point every run in this module at a private cache root."""
    directory = tmp_path / "cache"
    monkeypatch.setattr("stirrup.core.cache.DEFAULT_CACHE_DIR", directory)
    return directory


async def _interrupt_after_one_note(
    tools: list[Tool | LocalCodeExecToolProvider],
    *,
    input_files: Path | None = None,
    skills_dir: Path | None = None,
    system_prompt: str | None = None,
    model_slug: str = "identity-model",
    max_turns: int = 10,
) -> None:
    """Run one accepted tool turn, then fail, leaving a cache behind."""
    client = RecordingClient(
        [_note_turn("first", "call-1"), RuntimeError("interrupted")],
        model_slug=model_slug,
    )
    agent = _agent(client, tools, system_prompt=system_prompt, max_turns=max_turns)
    with pytest.raises(RuntimeError, match="interrupted"):
        async with agent.session(input_files=input_files, skills_dir=skills_dir) as session:
            await session.run(PROMPT)


async def _resume_to_finish(
    tools: list[Tool | LocalCodeExecToolProvider],
    *,
    input_files: Path | None = None,
    skills_dir: Path | None = None,
    system_prompt: str | None = None,
    model_slug: str = "identity-model",
    max_turns: int = 10,
) -> tuple[RecordingClient, RecordingLogger]:
    """Attempt a resume of the same task, finishing on the first turn it is given."""
    client = RecordingClient([_finish_turn("done", "call-finish")], model_slug=model_slug)
    logger = RecordingLogger()
    agent = _agent(client, tools, system_prompt=system_prompt, max_turns=max_turns, logger=logger)
    async with agent.session(input_files=input_files, skills_dir=skills_dir, resume=True) as session:
        finish_params, _, _ = await session.run(PROMPT)
    assert finish_params is not None
    return client, logger


def _write_skill(skills_dir: Path, body: str) -> None:
    skill_dir = skills_dir / "analysis"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"---\nname: analysis\ndescription: Analyze data.\n---\n\n{body}\n")


async def test_unchanged_identity_resumes_without_repeating_accepted_tool_calls() -> None:
    await _interrupt_after_one_note([_note_tool([])])

    resumed_notes: list[str] = []
    client, logger = await _resume_to_finish([_note_tool(resumed_notes)])

    assert _was_resumed(client)
    assert _kept_note_result(client, "first")
    assert resumed_notes == []
    assert any("Resuming from cached state" in message for message in logger.messages)


async def test_editing_an_input_file_starts_a_fresh_run(tmp_path: Path) -> None:
    input_file = tmp_path / "data.csv"
    input_file.write_text("a,b\n1,2\n")
    await _interrupt_after_one_note(
        [LocalCodeExecToolProvider(), _note_tool([])],
        input_files=input_file,
    )

    input_file.write_text("a,b\n9,9\n")
    resumed_notes: list[str] = []
    client, _ = await _resume_to_finish(
        [LocalCodeExecToolProvider(), _note_tool(resumed_notes)],
        input_files=input_file,
    )

    assert not _was_resumed(client)


async def test_unedited_input_file_still_resumes(tmp_path: Path) -> None:
    input_file = tmp_path / "data.csv"
    input_file.write_text("a,b\n1,2\n")
    await _interrupt_after_one_note(
        [LocalCodeExecToolProvider(), _note_tool([])],
        input_files=input_file,
    )

    client, _ = await _resume_to_finish(
        [LocalCodeExecToolProvider(), _note_tool([])],
        input_files=input_file,
    )

    assert _was_resumed(client)


async def test_editing_a_skill_body_starts_a_fresh_run(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    _write_skill(skills_dir, "Read every column before summarizing.")
    await _interrupt_after_one_note(
        [LocalCodeExecToolProvider(), _note_tool([])],
        skills_dir=skills_dir,
    )

    # Only the body changes: the name and description the system prompt shows are identical.
    _write_skill(skills_dir, "Ignore every column except the first.")
    client, _ = await _resume_to_finish(
        [LocalCodeExecToolProvider(), _note_tool([])],
        skills_dir=skills_dir,
    )

    assert not _was_resumed(client)


async def test_changing_a_tool_description_starts_a_fresh_run() -> None:
    await _interrupt_after_one_note([_note_tool([], description="Record a note")])

    client, _ = await _resume_to_finish([_note_tool([], description="Record a note, then stop")])

    assert not _was_resumed(client)


async def test_removing_a_tool_starts_a_fresh_run() -> None:
    await _interrupt_after_one_note([_note_tool([])])

    client, _ = await _resume_to_finish([])

    assert not _was_resumed(client)


async def test_changing_the_model_starts_a_fresh_run() -> None:
    await _interrupt_after_one_note([_note_tool([])], model_slug="identity-model")

    client, _ = await _resume_to_finish([_note_tool([])], model_slug="a-different-model")

    assert not _was_resumed(client)


async def test_changing_the_system_prompt_starts_a_fresh_run() -> None:
    await _interrupt_after_one_note([_note_tool([])], system_prompt="Be terse.")

    client, _ = await _resume_to_finish([_note_tool([])], system_prompt="Be exhaustive.")

    assert not _was_resumed(client)


async def test_raising_max_turns_still_resumes() -> None:
    # max_turns is deliberately excluded from identity: raising it and resuming is the reason
    # a max-turns cache exists at all.
    client = RecordingClient([_note_turn("first", "call-1")])
    agent = _agent(client, [_note_tool([])], max_turns=1)
    async with agent.session() as session:
        finish_params, _, _ = await session.run(PROMPT)
    assert finish_params is None

    resumed_notes: list[str] = []
    resumed_client, logger = await _resume_to_finish([_note_tool(resumed_notes)], max_turns=5)

    assert _was_resumed(resumed_client)
    assert any("Resuming from cached state" in message for message in logger.messages)
    # The turn that exhausted max_turns is part of the cache, so its tool result is restored
    # rather than produced again.
    assert _kept_note_result(resumed_client, "first")
    assert resumed_notes == []


async def test_a_corrupt_cache_is_reported_as_rejected_rather_than_absent(cache_base_dir: Path) -> None:
    await _interrupt_after_one_note([_note_tool([])])
    (_cache_dir(cache_base_dir) / "state.json").write_text('{"msgs": [{"role": "user"')

    client, logger = await _resume_to_finish([_note_tool([])])

    assert not _was_resumed(client)
    assert any("cannot be resumed" in message for message in logger.messages)
    assert any("will be made again" in message for message in logger.messages)
    assert not any("No cache found" in message for message in logger.messages)


async def test_a_malformed_transcript_is_reported_as_rejected(cache_base_dir: Path) -> None:
    await _interrupt_after_one_note([_note_tool([])])
    state_file = _cache_dir(cache_base_dir) / "state.json"
    state = json.loads(state_file.read_text())
    state["msgs"] = "not a list of messages"
    state_file.write_text(json.dumps(state))

    client, logger = await _resume_to_finish([_note_tool([])])

    assert not _was_resumed(client)
    assert any("cannot be resumed" in message for message in logger.messages)


async def test_malformed_run_metadata_is_reported_as_rejected(cache_base_dir: Path) -> None:
    await _interrupt_after_one_note([_note_tool([])])
    state_file = _cache_dir(cache_base_dir) / "state.json"
    state = json.loads(state_file.read_text())
    state["run_metadata_by_turn"] = "corrupt"
    state_file.write_text(json.dumps(state))

    client, logger = await _resume_to_finish([_note_tool([])])

    assert not _was_resumed(client)
    assert any("cannot be resumed" in message for message in logger.messages)


async def test_a_cache_missing_its_files_is_reported_as_rejected(cache_base_dir: Path, tmp_path: Path) -> None:
    input_file = tmp_path / "data.csv"
    input_file.write_text("a,b\n1,2\n")
    await _interrupt_after_one_note(
        [LocalCodeExecToolProvider(), _note_tool([])],
        input_files=input_file,
    )

    shutil.rmtree(_cache_dir(cache_base_dir) / "files")

    client, logger = await _resume_to_finish(
        [LocalCodeExecToolProvider(), _note_tool([])],
        input_files=input_file,
    )

    assert not _was_resumed(client)
    assert any("cannot be resumed" in message for message in logger.messages)


async def test_an_uncommitted_cache_is_reported_as_rejected(cache_base_dir: Path) -> None:
    await _interrupt_after_one_note([_note_tool([])])
    (_cache_dir(cache_base_dir) / "state.json").unlink()

    client, logger = await _resume_to_finish([_note_tool([])])

    assert not _was_resumed(client)
    assert any("cannot be resumed" in message for message in logger.messages)


async def test_no_cache_at_all_is_reported_as_absent() -> None:
    client, logger = await _resume_to_finish([_note_tool([])])

    assert not _was_resumed(client)
    assert any("No cache found" in message for message in logger.messages)
    assert not any("cannot be resumed" in message for message in logger.messages)


async def test_unreadable_input_files_disable_caching_without_failing_the_run(
    cache_base_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_file = tmp_path / "data.csv"
    input_file.write_text("a,b\n1,2\n")

    async def unreadable(_self: LocalCodeExecToolProvider, path: str) -> bytes:
        raise OSError(f"cannot read {path}")

    monkeypatch.setattr(LocalCodeExecToolProvider, "read_file_bytes", unreadable)

    client = RecordingClient([_note_turn("first", "call-1"), RuntimeError("interrupted")])
    logger = RecordingLogger()
    agent = _agent(client, [LocalCodeExecToolProvider(), _note_tool([])], logger=logger)
    with pytest.raises(RuntimeError, match="interrupted"):
        async with agent.session(input_files=input_file) as session:
            await session.run(PROMPT)

    assert any("caching disabled for this run" in message for message in logger.messages)
    assert not cache_base_dir.exists()


async def test_a_subagent_run_completes_without_a_cache_identity() -> None:
    child = Agent(
        client=RecordingClient([_finish_turn("child done", "call-child")]),
        name="child",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=AgentLogger(show_spinner=False),
    )
    parent_client = RecordingClient(
        [
            AssistantMessage(
                content="delegating",
                tool_calls=[
                    ToolCall(name="child", arguments=json.dumps({"task": "do the work"}), tool_call_id="call-sub")
                ],
                token_usage=TokenUsage(),
            ),
            _finish_turn("parent done", "call-parent"),
        ]
    )
    parent = _agent(parent_client, [child.to_tool()])

    async with parent.session() as session:
        finish_params, _, _ = await session.run(PROMPT)

    assert finish_params is not None
    assert finish_params.reason == "parent done"


def test_identical_bytes_at_different_destinations_have_distinct_identities() -> None:
    digest = "0" * 64
    one = compute_task_hash(PROMPT, input_files=[("/workspace/a/x.csv", digest)])
    other = compute_task_hash(PROMPT, input_files=[("/workspace/b/x.csv", digest)])

    assert one != other


def test_a_prompt_never_shares_an_identity_with_its_own_serialization() -> None:
    messages: list[ChatMessage] = [UserMessage(content=PROMPT)]
    serialized = json.dumps([message.model_dump(mode="json") for message in messages], sort_keys=True)

    assert compute_task_hash(PROMPT) != compute_task_hash(messages)
    assert compute_task_hash(serialized) != compute_task_hash(messages)
