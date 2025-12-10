"""Prompt templates for agent framework.

All prompts are loaded at module initialization from the prompts directory.
Templates can be formatted using .format() with the appropriate variables.
"""

from importlib.resources import files

prompts_dir = files("stirrup.prompts")

# Templates that need .format() with runtime values
MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE = (prompts_dir / "message_summarizer_bridge.txt").read_text(encoding="utf-8")
BASE_SYSTEM_PROMPT_TEMPLATE = (prompts_dir / "base_system_prompt.txt").read_text(encoding="utf-8")

# Ready-to-use prompts (no formatting needed)
MESSAGE_SUMMARIZER = (prompts_dir / "message_summarizer.txt").read_text(encoding="utf-8")

__all__ = [
    "BASE_SYSTEM_PROMPT_TEMPLATE",
    "MESSAGE_SUMMARIZER",
    "MESSAGE_SUMMARIZER_BRIDGE_TEMPLATE",
]
