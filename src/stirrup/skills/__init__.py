"""Skills module for agent capabilities.

This module provides functionality for loading and managing agent skills.
Skills are modular packages with instructions and resources that agents
can discover and use dynamically.

Example usage:
    from stirrup.skills import load_skills_metadata, format_skills_section
    from pathlib import Path

    # Load skills from directory
    skills = load_skills_metadata(Path("skills"))

    # Format for system prompt
    prompt_section = format_skills_section(skills)
"""

from stirrup.skills.skills import SkillMetadata, format_skills_section, load_skills_metadata

__all__ = [
    "SkillMetadata",
    "format_skills_section",
    "load_skills_metadata",
]
