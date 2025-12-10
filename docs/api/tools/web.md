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

::: stirrup.tools.web.FetchWebPageParams

## Web Search Tool

Searches the web using the Brave Search API.

!!! note
    Requires `BRAVE_API_KEY` environment variable.

::: stirrup.tools.web.WebSearchParams
