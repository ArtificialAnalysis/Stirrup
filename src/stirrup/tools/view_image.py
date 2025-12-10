"""View image tool provider for execution environments."""

from stirrup.core.models import Tool, ToolProvider, ToolUseCountMetadata
from stirrup.tools.code_backends.base import CodeExecToolProvider, ViewImageParams


class ViewImageToolProvider(ToolProvider):
    """Tool provider for viewing images from an execution environment.

    Can be used with an explicit exec_env or will auto-detect from the
    Agent's session state. Works regardless of tool ordering in the Agent.

    Examples:
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        client = ChatCompletionsClient(model="gpt-5")

        # Explicit exec_env
        exec_env = LocalCodeExecToolProvider()
        agent = Agent(
            client=client, name="assistant",
            tools=[exec_env, ViewImageToolProvider(exec_env)],
        )

        # Auto-detect (any order works)
        agent = Agent(
            client=client, name="assistant",
            tools=[ViewImageToolProvider(), LocalCodeExecToolProvider()],
        )

    """

    def __init__(
        self,
        exec_env: CodeExecToolProvider | None = None,
        *,
        name: str = "view_image",
        description: str | None = None,
    ) -> None:
        """Initialize ViewImageToolProvider.

        Args:
            exec_env: Optional execution environment. If None, will auto-detect
                from the Agent's session state.
            name: Tool name (default: "view_image").
            description: Tool description (default: standard description).

        """
        self._exec_env = exec_env
        self._name = name
        self._description = description

    async def __aenter__(self) -> Tool[ViewImageParams, ToolUseCountMetadata]:
        """Enter async context: resolve exec_env and return view_image tool."""
        # Import here to avoid circular dependency
        from stirrup.core.agent import _SESSION_STATE

        state = _SESSION_STATE.get(None)
        agent_exec_env = state.exec_env if state else None

        if self._exec_env is not None:
            # Explicit exec_env provided - validate it matches agent's exec_env
            if agent_exec_env is not None and self._exec_env is not agent_exec_env:
                raise ValueError(
                    f"ViewImageToolProvider exec_env ({type(self._exec_env).__name__}) "
                    f"does not match Agent's exec_env ({type(agent_exec_env).__name__}). "
                    "Use the same exec_env instance or omit exec_env to auto-detect."
                )
            exec_env = self._exec_env
        else:
            # Auto-detect from session state
            if agent_exec_env is None:
                raise RuntimeError(
                    "ViewImageToolProvider requires a CodeExecToolProvider. "
                    "Either pass exec_env explicitly or include a CodeExecToolProvider "
                    "in the Agent's tools list."
                )
            exec_env = agent_exec_env

        return exec_env.get_view_image_tool(
            name=self._name,
            description=self._description,
        )
