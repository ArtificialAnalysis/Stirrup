# Tool naming
FINISH_TOOL_NAME = "finish"

# Agent execution limits
AGENT_MAX_TURNS = 30  # Maximum agent turns before forced termination
CONTEXT_SUMMARIZATION_CUTOFF = 0.7  # Context window usage threshold (0.0-1.0) that triggers message summarization

# Media resolution limits
RESOLUTION_1MP = 1_000_000  # 1 megapixel - default max resolution for images
RESOLUTION_480P = 640 * 480  # 480p video resolution

# Code execution
SUBMISSION_SANDBOX_TIMEOUT = 60 * 10  # 10 minutes
E2B_SANDBOX_TEMPLATE_ALIAS = "e2b-sandbox"
