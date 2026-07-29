# Exceptions

`ContextOverflowError` means the request input does not fit the model context,
so `Agent` may shorten the conversation and retry. `OutputTokenLimitError` means
the provider exhausted the configured `max_tokens` response budget. An output
limit failure surfaces without summarization or retry with the unchanged limit.
`IncompleteResponseError` means the provider stopped for some other reason, such as
a content filter; it also surfaces without retry.

::: stirrup.core.exceptions
