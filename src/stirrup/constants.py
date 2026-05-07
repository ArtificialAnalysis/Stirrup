import warnings
from typing import Any, Literal

# Tool naming — the conventional name for the default finish tool (SIMPLE_FINISH_TOOL).
# Custom finish tools may use any name; only the default is bound to this constant.
DEFAULT_FINISH_TOOL_NAME: Literal["finish"] = "finish"


def __getattr__(name: str) -> Any:  # noqa: ANN401
    if name == "FINISH_TOOL_NAME":
        warnings.warn(
            "FINISH_TOOL_NAME is deprecated; use DEFAULT_FINISH_TOOL_NAME instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return DEFAULT_FINISH_TOOL_NAME
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Agent execution limits
AGENT_MAX_TURNS = 30  # Maximum agent turns before forced termination
CONTEXT_SUMMARIZATION_CUTOFF = 0.7  # Context window usage threshold (0.0-1.0) that triggers message summarization
TURNS_REMAINING_WARNING_THRESHOLD = 20

# Media resolution limits
RESOLUTION_1MP = 1_000_000  # 1 megapixel - default max resolution for images
RESOLUTION_480P = 640 * 480  # 480p video resolution

# Code execution
SANDBOX_TIMEOUT = 60 * 10  # 10 minutes
SANDBOX_REQUEST_TIMEOUT = 60 * 3  # 3 minutes
E2B_SANDBOX_TEMPLATE_ALIAS = "e2b-sandbox"
