# ChatCompletions Client

`max_tokens` limits provider output. `context_window_tokens` separately tells the
agent when conversation history should be summarized; when omitted, it defaults
to `max_tokens` for compatibility.

::: stirrup.clients.chat_completions_client
