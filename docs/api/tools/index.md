# Tools Module

The `stirrup.tools` module provides tools, tool providers, and the default tool set.

## DEFAULT_TOOLS

The standard set of tool providers included with every agent (unless overridden):

```python
from stirrup.tools import DEFAULT_TOOLS

# DEFAULT_TOOLS contains:
# - LocalCodeExecToolProvider() → provides "code_exec" tool
# - WebToolProvider() → provides "web_fetch" and "web_search" tools
```

## Built-in Tools

Simple tools that don't require lifecycle management:

::: stirrup.tools.calculator.CALCULATOR_TOOL

::: stirrup.tools.finish.SIMPLE_FINISH_TOOL

::: stirrup.tools.finish.FinishParams

## Tool Providers

Tool providers for lifecycle-managed tools:

- [Web Tools](web.md) - `WebToolProvider` for web fetch and search
- [Code Execution](code_backends.md) - `LocalCodeExecToolProvider`, `DockerCodeExecToolProvider`, `E2BCodeExecToolProvider`
- [Browser Use](browser-use.md) - `BrowserUseToolProvider` for browser automation
- [MCP](mcp.md) - `MCPToolProvider` for MCP server integration
- [View Image](view-image.md) - `ViewImageToolProvider` for viewing images
