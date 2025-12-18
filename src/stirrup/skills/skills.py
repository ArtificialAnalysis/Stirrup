"""Skills loader for agent capabilities.

Skills are modular packages that extend agent capabilities with instructions,
scripts, and resources. Each skill is a directory containing a SKILL.md file
with YAML frontmatter (name, description) and detailed instructions.

Based on: https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SkillMetadata:
    """Metadata extracted from a skill's SKILL.md frontmatter."""

    name: str
    description: str
    path: str  # Relative path like "skills/data_analysis"


def parse_frontmatter(content: str) -> dict[str, str]:
    """Parse YAML frontmatter from markdown content.

    Extracts metadata between --- markers at the start of the file.

    Args:
        content: Markdown content with optional YAML frontmatter

    Returns:
        Dictionary of frontmatter key-value pairs, empty if no frontmatter found

    """
    # Match YAML frontmatter between --- markers
    pattern = r"^---\s*\n(.*?)\n---"
    match = re.match(pattern, content, re.DOTALL)

    if not match:
        return {}

    frontmatter_text = match.group(1)
    result: dict[str, str] = {}

    # Simple YAML parsing for key: value pairs
    for line in frontmatter_text.strip().split("\n"):
        line = line.strip()
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()

    return result


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

    skills: list[SkillMetadata] = []

    for skill_path in skills_dir.iterdir():
        if not skill_path.is_dir():
            continue

        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            logger.debug("Skill directory missing SKILL.md: %s", skill_path)
            continue

        try:
            content = skill_md.read_text(encoding="utf-8")
            metadata = parse_frontmatter(content)

            name = metadata.get("name")
            description = metadata.get("description")

            if not name or not description:
                logger.warning(
                    "Skill %s missing required frontmatter (name, description)",
                    skill_path.name,
                )
                continue

            skills.append(
                SkillMetadata(
                    name=name,
                    description=description,
                    path=f"skills/{skill_path.name}",
                )
            )
            logger.debug("Loaded skill: %s", name)

        except Exception as e:
            logger.warning("Failed to load skill %s: %s", skill_path.name, e)

    return skills


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
        "1. Read the full instructions: `cat <skill_path>/SKILL.md`",
        "2. Follow the instructions and use any bundled resources as described",
        "",
    ]
    lines.extend([f"- **{skill.name}**: {skill.description} (`{skill.path}/SKILL.md`)" for skill in skills])

    return "\n".join(lines)
