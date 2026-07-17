# Exceptions

`ContextOverflowError` means the request input does not fit the model context,
so `Agent` may shorten the conversation and retry. `OutputTokenLimitError` means
the provider exhausted the configured `max_tokens` response budget. An output
limit failure surfaces without summarization or retry with the unchanged limit.

::: stirrup.core.exceptions
