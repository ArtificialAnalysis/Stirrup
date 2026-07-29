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

- The response body is not bounded while it is downloaded and decompressed; only the extracted
  markdown is truncated, to 40 000 characters. Bounding the download itself needs streaming reads
  plus a bounded incremental decompressor, which is out of scope here.
- There is no destination-port policy once an address validates: any port on a public address is
  reachable.
- Internal services hosted on public IP addresses remain reachable by construction.

::: stirrup.tools.web.FetchWebPageParams

## Web Search Tool

Searches the web using the Brave Search API.

!!! note
    Requires `BRAVE_API_KEY` environment variable.

::: stirrup.tools.web.WebSearchParams
