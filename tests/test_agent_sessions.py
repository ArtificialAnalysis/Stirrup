"""Behavioural contracts for per-session Agent runtime state."""

import json
from pathlib import Path

import anyio
import pytest

from stirrup.constants import DEFAULT_FINISH_TOOL_NAME
from stirrup.core.agent import Agent
from stirrup.core.cache import CacheManager, compute_task_hash
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    LLMClient,
    TokenUsage,
    Tool,
    ToolCall,
)
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider
from stirrup.tools.finish import SIMPLE_FINISH_TOOL
from stirrup.utils.logging import AgentLogger


def _quiet_logger() -> AgentLogger:
    return AgentLogger(show_spinner=False)


def _finish_message(reason: str, paths: list[str] | None = None) -> AssistantMessage:
    return AssistantMessage(
        content=reason,
        tool_calls=[
            ToolCall(
                name=DEFAULT_FINISH_TOOL_NAME,
                arguments=json.dumps({"reason": reason, "paths": paths or []}),
                tool_call_id=f"call-{reason}",
            )
        ],
        token_usage=TokenUsage(input=10, answer=5),
    )


class ScriptedClient(LLMClient):
    """Reply with one scripted message per call, recording what each call was given."""

    def __init__(self, responses: list[AssistantMessage]) -> None:
        self.responses = list(responses)
        self.tools_seen: list[set[str]] = []
        self.messages_seen: list[list[ChatMessage]] = []

    @property
    def model_slug(self) -> str:
        return "scripted-model"

    @property
    def max_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        self.tools_seen.append(set(tools))
        self.messages_seen.append(list(messages))
        return self.responses.pop(0)

    def system_prompt(self, call: int = 0) -> str:
        return str(self.messages_seen[call][0].content)


class GatedLocalProvider(LocalCodeExecToolProvider):
    """A backend whose first startup suspends, as the Docker, E2B and MCP providers do."""

    def __init__(self, suspended: anyio.Event, resume: anyio.Event) -> None:
        super().__init__()
        self._suspended = suspended
        self._resume = resume

    async def __aenter__(self) -> Tool:
        if not self._suspended.is_set():
            self._suspended.set()
            await self._resume.wait()
        return await super().__aenter__()


async def test_overlapping_session_cannot_strip_the_open_sessions_tools() -> None:
    """An open session keeps the tools it was given, whatever another session does."""
    client = ScriptedClient([_finish_message("done")])
    agent = Agent(
        client=client,
        name="one-agent",
        max_turns=2,
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async with agent.session(cache_on_interrupt=False) as session:
        assert {"code_exec", "fetch_web_page", DEFAULT_FINISH_TOOL_NAME} <= set(session.tools)

        # Assert on the toolset before the refusal, so a session that is wrongly allowed
        # reports the capability loss it caused rather than only the missing error.
        overlap_error: RuntimeError | None = None
        try:
            async with agent.session(cache_on_interrupt=False):
                pass
        except RuntimeError as exc:
            overlap_error = exc

        assert {"code_exec", "fetch_web_page", DEFAULT_FINISH_TOOL_NAME} <= set(session.tools)
        assert overlap_error is not None
        assert "Overlapping sessions cannot share configured" in str(overlap_error)

        finish_params, _, _ = await session.run("task")

    assert finish_params is not None
    assert {"code_exec", "fetch_web_page"} <= client.tools_seen[-1]


async def test_overlapping_session_is_refused_before_it_enters_the_shared_provider() -> None:
    """The refusal must land before the second session disturbs the first one's backend."""
    provider = LocalCodeExecToolProvider()
    agent = Agent(
        client=ScriptedClient([]),
        name="shared-provider",
        tools=[provider],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async with agent.session(cache_on_interrupt=False):
        entered_temp_dir = provider.temp_dir
        assert entered_temp_dir is not None

        with pytest.raises(RuntimeError, match="cannot share configured LocalCodeExecToolProvider"):
            async with agent.session(cache_on_interrupt=False):
                pass

        # A second entry would have replaced the temp directory the open session is using.
        assert provider.temp_dir is entered_temp_dir
        assert entered_temp_dir.exists()

    assert not entered_temp_dir.exists()


async def test_overlapping_provider_free_session_is_refused_over_the_shared_logger() -> None:
    """Every configured lifecycle object is reserved, not just the tool providers."""
    agent = Agent(
        client=ScriptedClient([]),
        name="provider-free",
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async with agent.session(cache_on_interrupt=False):
        with pytest.raises(RuntimeError, match="cannot share configured AgentLogger"):
            async with agent.session(cache_on_interrupt=False):
                pass

    # The reservation is released on exit, so sequential sessions stay supported.
    async with agent.session(cache_on_interrupt=False):
        pass


async def test_session_uploads_the_input_files_it_was_configured_with(tmp_path: Path) -> None:
    """Configuring a second session must not redirect the inputs of one still starting up."""
    (tmp_path / "alpha.txt").write_text("ALPHA")
    (tmp_path / "beta.txt").write_text("BETA")

    suspended = anyio.Event()
    resume = anyio.Event()
    alpha_finished = anyio.Event()
    client = ScriptedClient([_finish_message("alpha done"), _finish_message("beta done")])
    agent = Agent(
        client=client,
        name="input-files",
        max_turns=2,
        tools=[GatedLocalProvider(suspended, resume)],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async def run_alpha() -> None:
        async with agent.session(input_files=tmp_path / "alpha.txt", cache_on_interrupt=False) as session:
            await session.run("alpha task")
        alpha_finished.set()

    async def run_beta() -> None:
        # Prepared while alpha is suspended inside its backend's startup, entered afterwards.
        await suspended.wait()
        beta = agent.session(input_files=tmp_path / "beta.txt", cache_on_interrupt=False)
        resume.set()
        await alpha_finished.wait()
        async with beta as session:
            await session.run("beta task")

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_alpha)
        task_group.start_soon(run_beta)

    assert "alpha.txt" in client.system_prompt(0)
    assert "beta.txt" not in client.system_prompt(0)
    assert "beta.txt" in client.system_prompt(1)
    assert "alpha.txt" not in client.system_prompt(1)


async def test_interrupt_cache_is_filed_under_the_sessions_own_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run elsewhere on the same Agent must not redirect a session's interrupt cache."""
    monkeypatch.setattr("stirrup.core.cache.DEFAULT_CACHE_DIR", tmp_path)

    session_run_started = anyio.Event()
    plain_run_started = anyio.Event()

    class InterleavingClient(ScriptedClient):
        async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:  # noqa: ARG002
            if str(messages[-1].content) == "session task":
                session_run_started.set()
                await plain_run_started.wait()
                raise RuntimeError("session task interrupted")
            plain_run_started.set()
            raise RuntimeError("plain task failed")

    agent = Agent(
        client=InterleavingClient([]),
        name="cache-agent",
        max_turns=2,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async def run_in_session() -> None:
        with pytest.raises(RuntimeError, match="session task interrupted"):
            async with agent.session() as session:
                await session.run("session task")

    async def run_without_session() -> None:
        await session_run_started.wait()
        with pytest.raises(RuntimeError, match="plain task failed"):
            await agent.run("plain task")

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_in_session)
        task_group.start_soon(run_without_session)

    cache_manager = CacheManager(cache_base_dir=tmp_path)
    cached = cache_manager.load_state(compute_task_hash("session task"))
    assert cached is not None
    assert any(str(message.content) == "session task" for message in cached.msgs)
    assert cache_manager.load_state(compute_task_hash("plain task")) is None


async def test_nested_session_hands_the_outer_session_back_its_own_resources(tmp_path: Path) -> None:
    """Closing an inner session must not leave the outer one owning the inner one's state."""
    outer_provider = LocalCodeExecToolProvider()
    outer = Agent(
        client=ScriptedClient([_finish_message("outer done", ["outer.txt"])]),
        name="outer",
        max_turns=2,
        tools=[outer_provider],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    inner_provider = LocalCodeExecToolProvider()
    inner = Agent(
        client=ScriptedClient([_finish_message("inner done", ["inner.txt"])]),
        name="inner",
        max_turns=2,
        tools=[inner_provider],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )

    async with outer.session(output_dir=tmp_path / "outer-out", cache_on_interrupt=False) as outer_session:
        async with inner.session(output_dir=tmp_path / "inner-out", cache_on_interrupt=False) as inner_session:
            assert inner_session.tools["code_exec"] is not outer_session.tools["code_exec"]
            await inner_provider.write_file_bytes("inner.txt", b"INNER")
            await inner_session.run("inner task")

        outer_temp_dir = outer_provider.temp_dir
        await outer_provider.write_file_bytes("outer.txt", b"OUTER")
        finish_params, _, _ = await outer_session.run("outer task")

    assert finish_params is not None, "the outer finish tool validated against the inner session's backend"
    assert (tmp_path / "outer-out" / "outer.txt").read_bytes() == b"OUTER"
    assert sorted(path.name for path in (tmp_path / "inner-out").iterdir()) == ["inner.txt"]
    assert outer_temp_dir is not None and not outer_temp_dir.exists()


async def test_closed_session_does_not_leak_into_a_later_run(tmp_path: Path) -> None:
    """Once a session exits, nothing it owned may reach the next run in the same context."""
    (tmp_path / "secret.txt").write_text("SECRET")

    uploader = Agent(
        client=ScriptedClient([]),
        name="uploader",
        tools=[LocalCodeExecToolProvider()],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    async with uploader.session(input_files=tmp_path / "secret.txt", cache_on_interrupt=False):
        pass

    client = ScriptedClient([_finish_message("done", ["notes.txt"])])
    later = Agent(
        client=client,
        name="later",
        max_turns=2,
        tools=[],
        finish_tool=SIMPLE_FINISH_TOOL,
        logger=_quiet_logger(),
    )
    finish_params, _, _ = await later.run("later task")

    assert finish_params is not None, "the finish tool validated against the closed session's backend"
    assert "secret.txt" not in client.system_prompt()


async def test_two_agents_keep_their_own_files_while_running_concurrently(tmp_path: Path) -> None:
    """Separate Agents in separate tasks stay supported: the case sessions are built for."""
    both_running = anyio.Event()
    started: list[str] = []

    class RendezvousClient(ScriptedClient):
        async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:  # noqa: ARG002
            self.messages_seen.append(list(messages))
            started.append(self.model_slug)
            if len(started) == 2:
                both_running.set()
            await both_running.wait()
            return self.responses.pop(0)

    agents: dict[str, Agent] = {}
    for name in ("first", "second"):
        (tmp_path / f"{name}.txt").write_text(name.upper())
        agents[name] = Agent(
            client=RendezvousClient([_finish_message(f"{name} done", [f"{name}.txt"])]),
            name=name,
            max_turns=2,
            tools=[LocalCodeExecToolProvider()],
            finish_tool=SIMPLE_FINISH_TOOL,
            logger=_quiet_logger(),
        )

    async def run_agent(name: str) -> None:
        async with agents[name].session(
            input_files=tmp_path / f"{name}.txt",
            output_dir=tmp_path / f"{name}-out",
            cache_on_interrupt=False,
        ) as session:
            finish_params, _, _ = await session.run(f"{name} task")
        assert finish_params is not None

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(run_agent, "first")
        task_group.start_soon(run_agent, "second")

    assert (tmp_path / "first-out" / "first.txt").read_bytes() == b"FIRST"
    assert (tmp_path / "second-out" / "second.txt").read_bytes() == b"SECOND"
