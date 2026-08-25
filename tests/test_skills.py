"""Tests for Agent Skills parsing, discovery, and session integration."""

import logging
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from stirrup.core.agent import Agent, SubAgentParams
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolResult,
)
from stirrup.skills import SkillMetadata, format_skills_section, load_skills_metadata
from stirrup.skills.skills import parse_frontmatter
from stirrup.tools.code_backends.base import UploadFilesResult
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider


class RecordingClient:
    """Minimal client that records prompts and returns configured responses."""

    def __init__(self, responses: list[AssistantMessage] | None = None) -> None:
        self.responses = list(responses or [])
        self.messages_seen: list[list[ChatMessage]] = []

    @property
    def model_slug(self) -> str:
        return "recording-client"

    @property
    def context_window_tokens(self) -> int:
        return 100_000

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:  # noqa: ARG002
        self.messages_seen.append(list(messages))
        if not self.responses:
            raise AssertionError("Unexpected model generation")
        return self.responses.pop(0)


def write_skill(skills_dir: Path, directory_name: str, frontmatter: str, body: str = "# Instructions") -> Path:
    """Create a skill package and return its directory."""
    skill_dir = skills_dir / directory_name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n\n{body}\n", encoding="utf-8")
    return skill_dir


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    path = tmp_path / "skills"
    path.mkdir()
    return path


def test_parse_frontmatter_supports_yaml_scalars_and_nested_metadata() -> None:
    parsed = parse_frontmatter(
        """---
name: "pdf-processing"
description: >-
  Extracts text from PDFs.
  Use when handling PDF documents.
metadata:
  author: example-org
  version: "1.0"
---
# PDF processing
"""
    )

    assert parsed == {
        "name": "pdf-processing",
        "description": "Extracts text from PDFs. Use when handling PDF documents.",
        "metadata": {"author": "example-org", "version": "1.0"},
    }


def test_load_skills_preserves_optional_fields(skills_dir: Path) -> None:
    write_skill(
        skills_dir,
        "pdf-processing",
        """name: pdf-processing
description: Extract PDFs. Use when handling PDF documents.
license: Apache-2.0
compatibility: Requires Python 3.12+
metadata:
  author: example-org
  version: "1.0"
allowed-tools: Bash(python:*) Read""",
    )

    assert load_skills_metadata(skills_dir) == [
        SkillMetadata(
            name="pdf-processing",
            description="Extract PDFs. Use when handling PDF documents.",
            path="skills/pdf-processing",
            license="Apache-2.0",
            compatibility="Requires Python 3.12+",
            metadata={"author": "example-org", "version": "1.0"},
            allowed_tools="Bash(python:*) Read",
        )
    ]


@pytest.mark.parametrize(
    ("directory_name", "frontmatter", "error"),
    [
        (
            "pdf_processing",
            "name: pdf_processing\ndescription: Process PDFs. Use for PDF documents.",
            "only lowercase letters, numbers, and single hyphens",
        ),
        (
            "pdf--processing",
            "name: pdf--processing\ndescription: Process PDFs. Use for PDF documents.",
            "only lowercase letters, numbers, and single hyphens",
        ),
        (
            "a" * 65,
            f"name: {'a' * 65}\ndescription: Use for long-name validation.",
            "must be at most 64 characters",
        ),
        (
            "pdf-processing",
            "name: another-skill\ndescription: Process PDFs. Use for PDF documents.",
            "must match parent directory",
        ),
        (
            "pdf-processing",
            'name: pdf-processing\ndescription: ""',
            "must be a non-empty string",
        ),
        (
            "pdf-processing",
            f"name: pdf-processing\ndescription: {'x' * 1_025}",
            "must be at most 1024 characters",
        ),
        (
            "pdf-processing",
            "name: pdf-processing\ndescription: Process PDFs.\nmetadata:\n  version: 1",
            "must map strings to strings",
        ),
    ],
)
def test_invalid_skills_are_skipped_with_clear_diagnostics(
    skills_dir: Path,
    caplog: pytest.LogCaptureFixture,
    directory_name: str,
    frontmatter: str,
    error: str,
) -> None:
    write_skill(skills_dir, directory_name, frontmatter)

    with caplog.at_level(logging.WARNING):
        skills = load_skills_metadata(skills_dir)

    assert skills == []
    assert "Skipping invalid skill" in caplog.text
    assert error in caplog.text


def test_malformed_yaml_is_skipped_with_clear_diagnostic(
    skills_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_skill(
        skills_dir,
        "pdf-processing",
        "name: pdf-processing\ndescription: [unterminated",
    )

    with caplog.at_level(logging.WARNING):
        skills = load_skills_metadata(skills_dir)

    assert skills == []
    assert "Skipping invalid skill" in caplog.text
    assert "while parsing a flow sequence" in caplog.text


def test_skills_are_sorted_by_name(skills_dir: Path) -> None:
    write_skill(skills_dir, "zulu-skill", "name: zulu-skill\ndescription: Use for Zulu tasks.")
    write_skill(skills_dir, "alpha-skill", "name: alpha-skill\ndescription: Use for alpha tasks.")

    assert [skill.name for skill in load_skills_metadata(skills_dir)] == ["alpha-skill", "zulu-skill"]


def test_duplicate_names_are_skipped_deterministically(
    skills_dir: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_skill(skills_dir, "first", "name: first\ndescription: First.")
    write_skill(skills_dir, "second", "name: second\ndescription: Second.")

    def duplicate_metadata(skill_path: Path, _frontmatter: object) -> SkillMetadata:
        return SkillMetadata(
            name="duplicate",
            description=f"Loaded from {skill_path.name}",
            path=f"skills/{skill_path.name}",
        )

    monkeypatch.setattr("stirrup.skills.skills._skill_metadata", duplicate_metadata)
    with caplog.at_level(logging.WARNING):
        skills = load_skills_metadata(skills_dir)

    assert skills == [SkillMetadata(name="duplicate", description="Loaded from first", path="skills/first")]
    assert "Skipping duplicate skill name 'duplicate'" in caplog.text


def test_format_skills_section_explains_activation_and_path_resolution() -> None:
    section = format_skills_section(
        [SkillMetadata(name="pdf-processing", description="Use for PDFs.", path="skills/pdf-processing")]
    )

    assert "read its full instructions before proceeding" in section
    assert "Resolve relative paths against the skill directory" in section
    assert "skills/pdf-processing/SKILL.md" in section


async def test_skills_require_code_execution(skills_dir: Path) -> None:
    write_skill(skills_dir, "pdf-processing", "name: pdf-processing\ndescription: Use for PDFs.")
    agent = Agent(client=RecordingClient(), name="agent-without-code-exec", tools=[])

    with pytest.raises(ValueError, match="skills_dir specified but no CodeExecToolProvider"):
        async with agent.session(skills_dir=skills_dir, cache_on_interrupt=False):
            pass


async def test_skill_upload_failures_abort_session(
    skills_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_skill(skills_dir, "pdf-processing", "name: pdf-processing\ndescription: Use for PDFs.")
    provider = LocalCodeExecToolProvider()
    monkeypatch.setattr(
        provider,
        "upload_files",
        AsyncMock(return_value=UploadFilesResult(failed={str(skills_dir): "upload failed"})),
    )
    agent = Agent(client=RecordingClient(), name="upload-failure-agent", tools=[provider])

    with pytest.raises(RuntimeError, match="Failed to upload skills"):
        async with agent.session(skills_dir=skills_dir, cache_on_interrupt=False):
            pass


async def test_session_uploads_skills_and_builds_catalog(skills_dir: Path) -> None:
    write_skill(skills_dir, "pdf-processing", "name: pdf-processing\ndescription: Use for PDFs.")
    provider = LocalCodeExecToolProvider()
    agent = Agent(client=RecordingClient(), name="skills-agent", tools=[provider])

    async with agent.session(skills_dir=skills_dir, cache_on_interrupt=False) as session:
        prompt = session._build_system_prompt()  # noqa: SLF001
        uploaded = await provider.file_exists("skills/pdf-processing/SKILL.md")

    assert uploaded
    assert "**pdf-processing**: Use for PDFs." in prompt
    assert "skills/pdf-processing/SKILL.md" in prompt


async def test_subagent_inherits_skill_catalog_and_files(
    skills_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_skill(skills_dir, "pdf-processing", "name: pdf-processing\ndescription: Use for PDFs.")
    child_response = AssistantMessage(
        blocks=[
            ToolCall(
                name="finish",
                arguments='{"reason": "done", "paths": []}',
                tool_call_id="finish-call",
            )
        ],
        token_usage=TokenUsage(),
    )
    child_client = RecordingClient([child_response])
    child_provider = LocalCodeExecToolProvider()
    child_upload = AsyncMock(wraps=child_provider.upload_files)
    monkeypatch.setattr(child_provider, "upload_files", child_upload)
    child = Agent(client=child_client, name="skills-child", tools=[child_provider])
    child_tool = child.to_tool()

    parent_provider = LocalCodeExecToolProvider()
    parent = Agent(
        client=RecordingClient(),
        name="skills-parent",
        tools=[parent_provider, child_tool],
    )

    async with parent.session(skills_dir=skills_dir, cache_on_interrupt=False):
        execution = child_tool.executor(SubAgentParams(task="Use the relevant skill", input_files=[]))
        assert not isinstance(execution, ToolResult)
        result = await execution

    assert result.success
    child_upload.assert_awaited_once_with("skills", source_env=parent_provider)
    system_message = child_client.messages_seen[0][0]
    assert isinstance(system_message, SystemMessage)
    assert "skills/pdf-processing/SKILL.md" in system_message.content
