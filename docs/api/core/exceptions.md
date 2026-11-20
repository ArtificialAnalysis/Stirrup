# Exceptions

`ContextOverflowError` means the request input does not fit the model context,
so `Agent` drops the latest turn, summarizes the rest and retries the step once. `OutputTokenLimitError` means
the provider exhausted the configured `max_tokens` response budget. An output
limit failure surfaces without summarization or retry with the unchanged limit.
`IncompleteResponseError` means the provider stopped for some other reason, such as
a content filter; it also surfaces without retry. `EmptyResponseError` means
`ChatCompletionsClient` got replies without choices on every attempt, as gateways such
as OpenRouter send with HTTP 200 when an upstream provider fails; the client retries
such replies up to `max_retries` times before raising it, or raises it at once when
the error's code is a 4xx other than 408 or 429. `CompactionRequiredError` means the
provider rejected the request for a limit compaction can fix, too many images
(`reason="image_count"`) or too many request bytes (`reason="request_size"`); `Agent`
drops the latest turn, summarizes the rest and retries the step once.

::: stirrup.core.exceptions
