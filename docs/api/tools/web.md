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

### One approved Parallel documentation lookup

For an attended run, a host application can explicitly select Parallel Unified MCP
instead of Brave. Install `stirrup[mcp]` and obtain the user's approval for one literal
public documentation phrase before constructing the provider:

```python
from stirrup import Agent
from stirrup.tools import WebToolProvider

# The host obtains approval for this exact public phrase outside the model.
approved_phrase = "Python 3.13 tomllib TOMLDecodeError"
agent = Agent(
    client=client,  # Your configured LLMClient
    name="documentation_lookup",
    tools=[WebToolProvider(
        search_provider="parallel",
        parallel_public_query=approved_phrase,
        timeout=30,
    )],
)
async with agent.session() as session:
    finish, history, metadata = await session.run(
        "Use the approved documentation lookup to explain invalid TOML handling. "
        "Cite the original documentation URL returned by web_search."
    )
```

The argument-free `web_search` sends the exact phrase as its only search query to
`https://mcp.parallel.ai/mcp`, with a fixed objective to find official public
documentation. It sends no conversation, files, model identifier, API key, or prior
session identifier. No Parallel key is needed for the server's rate-limited anonymous
search. See [Parallel MCP setup](https://docs.parallel.ai/integrations/mcp/quickstart).

The host is responsible for confirming that the phrase is public and approved; the
provider cannot infer consent or detect private information in arbitrary text. Do not
derive this setting from a model response, private prompt, or background task. The
model can invoke the lookup with `{}`; extra arguments fail validation before dispatch.
Each provider instance allows one attempt, including failures, timeouts and cancellation.
Repeated calls, `get_tools()`, or re-entering that instance do not grant another attempt.
Construct a new provider only for a new user-approved operation.

Results retain original source URLs and excerpts for the agent's next turn. Empty
responses are reported to the agent; tool, connection and timeout errors return an
unsuccessful result without retries or provider fallback. The provider's `timeout`
bounds the complete MCP operation, including connection setup and cleanup. Web fetching
remains available with its existing behavior. Parallel is never added to `default_tools()`
or the Slack integration; absent explicit selection, Brave behavior is unchanged.
