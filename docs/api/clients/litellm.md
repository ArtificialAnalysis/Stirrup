# LiteLLM Client

!!! note "Optional Dependency"
    LiteLLM is an optional dependency. Install with:
    ```bash
    pip install stirrup[litellm]  # or: uv add stirrup[litellm]
    ```

`max_tokens` limits provider output. `context_window_tokens` (required) separately
tells the agent when conversation history should be summarized.

::: stirrup.clients.litellm_client
