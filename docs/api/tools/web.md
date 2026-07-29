# Web Tools

The `WebToolProvider` provides web fetching and search capabilities.

## WebToolProvider

::: stirrup.tools.web.WebToolProvider
    options:
      show_source: true
      members:
        - __init__
        - __aenter__
        - __aexit__

## Web Fetch Tool

Fetches a web page and returns its content as markdown.

!!! note "Fetch security"
    Web fetch resolves and validates every destination and redirect, rejects non-public
    addresses, ignores environment proxy settings, and neither sends nor retains cookies.
    Its transport and state are isolated from search, and provider cleanup is safe even
    when startup is interrupted.

### Limits

- Response bodies are read up to 1 MiB; longer pages are truncated before extraction and the
  returned markdown carries a truncation notice.
- Compressed responses are refused rather than decoded, because the decoder expands a whole chunk
  at a time and so cannot be bounded by the byte cap. If origins that ignore
  `Accept-Encoding: identity` show up in practice, the fix is a bounded incremental decompressor,
  not relaxing the check.
- There is no destination-port policy once an address validates: any port on a public address is
  reachable.
- Internal services hosted on public IP addresses remain reachable by construction.

::: stirrup.tools.web.FetchWebPageParams

## Web Search Tool

Searches the web using the Brave Search API.

!!! note
    Requires `BRAVE_API_KEY` environment variable.

::: stirrup.tools.web.WebSearchParams
