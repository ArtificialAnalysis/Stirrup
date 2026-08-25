"""Skills loader for agent capabilities.

Skills are modular packages that extend agent capabilities with instructions,
scripts, and resources. Each skill is a directory containing a SKILL.md file
with YAML frontmatter (name, description) and detailed instructions.

Based on: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_FRONTMATTER_PATTERN = re.compile(
    r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)",
    re.DOTALL,
)
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1_024
_MAX_COMPATIBILITY_LENGTH = 500


@dataclass
class SkillMetadata:
    """Metadata extracted from a skill's SKILL.md frontmatter."""

    name: str
    description: str
    path: str  # Runtime path like "skills/data-analysis"
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    allowed_tools: str | None = None


def parse_frontmatter(content: str) -> dict[str, Any]:
    """Parse YAML frontmatter from markdown content.

    Extracts metadata between --- markers at the start of the file.

    Args:
        content: Markdown content with optional YAML frontmatter

    Returns:
        Dictionary of frontmatter key-value pairs, empty if no frontmatter found

    Raises:
        yaml.YAMLError: If the frontmatter contains invalid YAML.
        ValueError: If the parsed frontmatter is not a mapping.

    """
    match = _FRONTMATTER_PATTERN.match(content)
    if not match:
        return {}

    parsed = yaml.safe_load(match.group(1))
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ValueError("frontmatter must be a YAML mapping")
    if not all(isinstance(key, str) for key in parsed):
        raise ValueError("frontmatter keys must be strings")
    return parsed


def _required_string(frontmatter: dict[str, Any], field_name: str, max_length: int) -> str:
    value = frontmatter.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"frontmatter field '{field_name}' must be a non-empty string")
    if len(value) > max_length:
        raise ValueError(f"frontmatter field '{field_name}' must be at most {max_length} characters")
    return value


def _optional_string(
    frontmatter: dict[str, Any],
    field_name: str,
    *,
    max_length: int | None = None,
) -> str | None:
    value = frontmatter.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"frontmatter field '{field_name}' must be a non-empty string when provided")
    if max_length is not None and len(value) > max_length:
        raise ValueError(f"frontmatter field '{field_name}' must be at most {max_length} characters")
    return value


def _skill_metadata(skill_path: Path, frontmatter: dict[str, Any]) -> SkillMetadata:
    name = _required_string(frontmatter, "name", _MAX_NAME_LENGTH)
    description = _required_string(frontmatter, "description", _MAX_DESCRIPTION_LENGTH)

    if not _SKILL_NAME_PATTERN.fullmatch(name):
        raise ValueError("frontmatter field 'name' must contain only lowercase letters, numbers, and single hyphens")
    if name != skill_path.name:
        raise ValueError(f"frontmatter field 'name' must match parent directory '{skill_path.name}'")

    raw_metadata = frontmatter.get("metadata")
    if raw_metadata is None:
        metadata: dict[str, str] = {}
    elif not isinstance(raw_metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in raw_metadata.items()
    ):
        raise ValueError("frontmatter field 'metadata' must map strings to strings")
    else:
        metadata = dict(raw_metadata)

    return SkillMetadata(
        name=name,
        description=description,
        path=f"skills/{skill_path.name}",
        license=_optional_string(frontmatter, "license"),
        compatibility=_optional_string(
            frontmatter,
            "compatibility",
            max_length=_MAX_COMPATIBILITY_LENGTH,
        ),
        metadata=metadata,
        allowed_tools=_optional_string(frontmatter, "allowed-tools"),
    )


def load_skills_metadata(skills_dir: Path) -> list[SkillMetadata]:
    """Scan skills directory for SKILL.md files and extract metadata.

    Args:
        skills_dir: Path to the skills directory

    Returns:
        List of SkillMetadata for each valid skill found.
        Returns empty list if skills_dir doesn't exist or has no skills.

    """
    if not skills_dir.exists():
        logger.debug("Skills directory does not exist: %s", skills_dir)
        return []

    if not skills_dir.is_dir():
        logger.warning("Skills path is not a directory: %s", skills_dir)
        return []

    skills_by_name: dict[str, SkillMetadata] = {}
    source_by_name: dict[str, Path] = {}

    skill_paths = sorted((path for path in skills_dir.iterdir() if path.is_dir()), key=lambda path: path.name)
    for skill_path in skill_paths:
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            logger.debug("Skill directory missing SKILL.md: %s", skill_path)
            continue

        try:
            content = skill_md.read_text(encoding="utf-8")
            skill = _skill_metadata(skill_path, parse_frontmatter(content))
            if skill.name in skills_by_name:
                logger.warning(
                    "Skipping duplicate skill name '%s' in %s; already loaded from %s",
                    skill.name,
                    skill_path,
                    source_by_name[skill.name],
                )
                continue

            skills_by_name[skill.name] = skill
            source_by_name[skill.name] = skill_path
            logger.debug("Loaded skill: %s", skill.name)
        except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
            logger.warning("Skipping invalid skill %s: %s", skill_path, exc)

    return sorted(skills_by_name.values(), key=lambda skill: skill.name)


def format_skills_section(skills: list[SkillMetadata]) -> str:
    """Format skills metadata as a system prompt section.

    Args:
        skills: List of skill metadata to include

    Returns:
        Formatted string for inclusion in system prompt.
        Returns empty string if no skills provided.

    """
    if not skills:
        return ""

    lines = [
        "## Available Skills",
        "",
        "You have access to the following skills located in the `skills/` directory. "
        "Each skill contains a SKILL.md file with detailed instructions and potentially bundled scripts.",
        "",
        "To use a skill:",
        "1. When a task matches a skill's description, read its full instructions before proceeding: "
        "`cat <skill_path>/SKILL.md`",
        "2. Resolve relative paths against the skill directory (`<skill_path>/`) and use the resolved path in tool calls",
        "3. Follow the instructions and use bundled resources only as described",
        "",
    ]
    lines.extend([f"- **{skill.name}**: {skill.description} (`{skill.path}/SKILL.md`)" for skill in skills])

    return "\n".join(lines)
