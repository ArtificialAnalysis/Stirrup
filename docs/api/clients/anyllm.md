# any-llm Client

!!! note "Optional Dependency"
    any-llm is an optional dependency. Install with:
    ```bash
    pip install stirrup[anyllm]  # or: uv add stirrup[anyllm]
    ```

`max_tokens` limits provider output. `context_window_tokens` (required) separately
tells the agent when conversation history should be summarized.

::: stirrup.clients.anyllm_client
