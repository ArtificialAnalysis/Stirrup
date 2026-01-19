# Browser Use Tool Provider

The `BrowserUseToolProvider` provides browser automation capabilities using the [browser-use](https://github.com/browser-use/browser-use) library.

## Installation

```bash
pip install 'stirrup[browser]'  # or: uv add 'stirrup[browser]'
```

## Prerequisites

**Local browser** (default):

```bash
# Install Chromium browser
uvx browser-use install
```

**Cloud browser** (optional):

For cloud-hosted browser sessions, set the `BROWSER_USE_API_KEY` environment variable:

```bash
export BROWSER_USE_API_KEY=your-api-key-here
```

Get your API key from the [Browser Use Cloud dashboard](https://cloud.browser-use.com).

## Quick Start

```python
--8<-- "examples/browser_use_example.py:main"
```

## Available Tools

When you add `BrowserUseToolProvider` to your agent, it exposes the following tools (all prefixed with `browser_` by default):

### Navigation

| Tool | Description |
|------|-------------|
| `browser_search` | Search using Google, DuckDuckGo, or Bing |
| `browser_navigate` | Navigate to a URL (optionally in new tab) |
| `browser_go_back` | Go back in browser history |
| `browser_wait` | Wait for specified seconds (1-30) |

### Page Interaction

| Tool | Description |
|------|-------------|
| `browser_click` | Click an element by index |
| `browser_input_text` | Type text into a form field |
| `browser_scroll` | Scroll up or down |
| `browser_find_text` | Find and scroll to specific text |
| `browser_send_keys` | Send keyboard keys (Enter, Tab, etc.) |

### JavaScript

| Tool | Description |
|------|-------------|
| `browser_evaluate_js` | Execute custom JavaScript code |

### Tab Management

| Tool | Description |
|------|-------------|
| `browser_switch_tab` | Switch to a different tab by index |

### Content Extraction

| Tool | Description |
|------|-------------|
| `browser_snapshot` | Get accessibility tree with element indices |
| `browser_screenshot` | Take a screenshot of the page |
| `browser_get_url` | Get current page URL |

## Workflow

The typical workflow for browser automation:

1. **Navigate** to a page using `browser_navigate` or `browser_search`
2. **Snapshot** the page using `browser_snapshot` to see elements and their indices
3. **Interact** with elements using `browser_click`, `browser_input_text`, etc.
4. **Repeat** snapshot and interaction as needed

The snapshot returns an accessibility tree showing interactive elements with numerical indices that you reference in other tools.

## Configuration

### BrowserUseToolProvider

::: stirrup.tools.browser_use.BrowserUseToolProvider
    options:
      show_source: true
      members:
        - __init__
        - __aenter__
        - __aexit__

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `headless` | `bool` | `True` | Run browser without visible window |
| `disable_security` | `bool` | `False` | Disable browser security features |
| `executable_path` | `str \| None` | `None` | Path to Chrome/Chromium |
| `cdp_url` | `str \| None` | `None` | CDP URL for remote browser |
| `tool_prefix` | `str` | `"browser"` | Prefix for tool names |
| `extra_args` | `list[str] \| None` | `None` | Extra Chromium args |

### Cloud Browser

To use a cloud-hosted browser:

```python
import os

browser_provider = BrowserUseToolProvider(
    use_cloud=True,  # Requires BROWSER_USE_API_KEY env var
)
```

This requires setting the `BROWSER_USE_API_KEY` environment variable.

## Example

See the full example at [`examples/browser_use_example.py`](https://github.com/ArtificialAnalysis/Stirrup/blob/main/examples/browser_use_example.py).

