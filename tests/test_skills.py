"""Tests for skill metadata loading."""

from pathlib import Path

import pytest

from stirrup.skills.skills import SkillMetadata, format_skills_section, load_skills_metadata, parse_frontmatter


@pytest.mark.parametrize(
    ("indicator", "expected"),
    [
        (">", "Inspect production metrics and identify likely causes."),
        ("|", "Inspect production metrics\nand identify likely causes."),
    ],
)
def test_parse_frontmatter_supports_yaml_block_scalars(indicator: str, expected: str) -> None:
    content = f"""---
name: investigate-metrics
description: {indicator}
  Inspect production metrics
  and identify likely causes.
---

# Instructions
"""

    assert parse_frontmatter(content) == {
        "name": "investigate-metrics",
        "description": expected,
    }


def test_parse_frontmatter_rejects_malformed_yaml() -> None:
    content = """---
name: [unterminated
description: broken
---
"""

    assert parse_frontmatter(content) == {}


def test_load_skills_metadata_reads_folded_description(tmp_path: Path) -> None:
    skill_dir = tmp_path / "investigate-metrics"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        """---
name: investigate-metrics
description: >
  Inspect production metrics and
  identify likely causes.
---
""",
        encoding="utf-8",
    )

    assert load_skills_metadata(tmp_path) == [
        SkillMetadata(
            name="investigate-metrics",
            description="Inspect production metrics and identify likely causes.",
            path="skills/investigate-metrics",
        )
    ]


def test_format_skills_section_does_not_assume_shell_access() -> None:
    section = format_skills_section(
        [SkillMetadata(name="investigate-metrics", description="Inspect metrics.", path="skills/investigate-metrics")]
    )

    assert "file-reading tool" in section
    assert "cat " not in section
