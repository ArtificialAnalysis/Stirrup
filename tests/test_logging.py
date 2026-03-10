"""Tests for AgentLogger tool result rendering."""

import logging
import signal
from unittest.mock import patch

from stirrup.core.models import ToolMessage
from stirrup.utils.logging import AgentLogger


def test_tool_result_does_not_hang_on_binary_content() -> None:
    """Binary tool output (e.g. `cat file.xlsx`) must render without hanging.

    With an XML lexer, Pygments' regex engine would backtrack indefinitely on
    binary data containing partial XML-like byte sequences. The text lexer
    avoids this entirely.
    """
    logger = AgentLogger(show_spinner=False, level=logging.DEBUG)
    logger.name = "test-agent"
    logger.depth = 0

    # Simulate binary file content — dense non-printable bytes with
    # interspersed < > & that would trigger XML lexer backtracking.
    binary_str = "".join(chr(b) for b in range(256)) * 10
    msg = ToolMessage(content=binary_str, name="bash", tool_call_id="call_1", args_was_valid=True)

    def _timeout(_signum: int, _frame: object) -> None:
        raise TimeoutError("tool_result hung on binary content — likely using XML lexer")

    prev = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(2)
    try:
        with patch("stirrup.utils.logging.console"):
            logger.tool_result(msg)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)
