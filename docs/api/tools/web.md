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
  markdown is truncated, to 40 000 characters. A compressed body is the sharper case, since it
  expands without bound and httpx decodes a whole chunk before any size could be measured.
  Refusing compressed responses outright was considered and rejected: it taxes every fetch with
  full uncompressed bandwidth and breaks origins that ignore `Accept-Encoding: identity`, while
  still leaving the uncompressed body unbounded. The bounded fix is streaming reads plus an
  incremental decompressor, which is out of scope here. Revisit if a fetch is observed to exhaust
  memory in practice.
- There is no destination-port policy once an address validates: any port on a public address is
  reachable.
- Internal services hosted on public IP addresses remain reachable by construction.

::: stirrup.tools.web.FetchWebPageParams

## Web Search Tool

Searches the web using the Brave Search API.

!!! note
    Requires `BRAVE_API_KEY` environment variable.

::: stirrup.tools.web.WebSearchParams

### Excluding domains

Search benchmarks often need some sites kept out of the results, for example the site that hosts
the benchmark's questions and answers. Pass `exclude_domains` to `WebToolProvider`, or set the
comma-separated `WEB_SEARCH_EXCLUDE_DOMAINS` environment variable so it also applies to
`default_tools()`:

```python
WebToolProvider(exclude_domains=["huggingface.co", "github.com"])
```

```bash
export WEB_SEARCH_EXCLUDE_DOMAINS="huggingface.co,github.com"
```

- An excluded domain also excludes its subdomains: `huggingface.co` removes `discuss.huggingface.co`.
- Entries may be bare domains or URLs (`https://huggingface.co/datasets` becomes `huggingface.co`).
  A leading `*.` is accepted and ignored.
- An explicit `exclude_domains` replaces the environment variable; `exclude_domains=[]` turns it off.
- Only `web_search` is filtered. `fetch_web_page` can still open a URL on an excluded domain.

Brave Search has no `exclude_domains` parameter. The provider sends one inline
[Goggle](https://api-dashboard.search.brave.com/documentation/resources/goggles) with a
`$discard,site=<domain>` rule per domain. Brave does not document whether `site=` covers
subdomains, so the tool also drops any returned result whose host is an excluded domain or one of
its subdomains, and counts those in `WebSearchMetadata.results_excluded`. A search can therefore
return fewer than 5 results.

Other search APIs name the same feature differently, which is the mapping to use when adding
another provider:

| Provider | Parameter |
| --- | --- |
| Brave Search | `goggles` with `$discard,site=<domain>` rules |
| Tavily | `exclude_domains` |
| Exa | `excludeDomains` |
| Parallel | `source_policy.exclude_domains` |
| Perplexity Search | `search_domain_filter`, each domain prefixed with `-` |
| Anthropic web search tool | `blocked_domains` |
| OpenAI web search tool | `filters.blocked_domains` |
| Vertex AI Google Search grounding | `exclude_domains` |
