# Custom Backends

This guide covers implementing custom code execution backends.

## CodeExecToolProvider

All code execution backends extend [`CodeExecToolProvider`][stirrup.tools.code_backends.CodeExecToolProvider]. Key methods to implement:

| Method | Purpose |
|--------|---------|
| `__aenter__` | Initialize environment and return `code_exec` tool |
| `__aexit__` | Cleanup environment (temp files, connections) |
| `run_command` | Execute a shell command and return `CommandResult` |
| `read_file_bytes` | Read file content from execution environment |
| `write_file_bytes` | Write file content to execution environment |

The base class provides:

- `get_code_exec_tool()` - Returns the standard `code_exec` tool
- `allowed_commands` - Optional regex patterns to restrict commands
- File upload/download utilities

## Minimal Implementation

```python
from stirrup.tools.code_backends import (
    CodeExecToolProvider,
    CommandResult,
    format_result,
)
from stirrup import Tool, ToolResult


class SimpleExecProvider(CodeExecToolProvider):
    """Simple execution in current directory."""

    async def __aenter__(self) -> Tool:
        return self.get_code_exec_tool()

    async def __aexit__(self, *args):
        pass  # No cleanup needed

    async def run_command(self, cmd: str, *, timeout: int = 300) -> CommandResult:
        import asyncio

        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )

            return CommandResult(
                exit_code=proc.returncode or 0,
                stdout=stdout.decode(),
                stderr=stderr.decode(),
            )

        except asyncio.TimeoutError:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr="",
                error_kind="timeout",
                advice=f"Command timed out after {timeout} seconds",
            )

    async def read_file_bytes(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        with open(path, "wb") as f:
            f.write(content)
```

## Command Allowlist

Restrict what commands can be executed:

```python
provider = MyCodeExecProvider(
    allowed_commands=[
        r"python.*",           # Allow Python commands
        r"pip install.*",      # Allow pip install
        r"ls.*",               # Allow ls
        r"cat.*",              # Allow cat
    ]
)
```

The base class validates commands before execution.

## Next Steps

- [Code Execution Guide](../guides/code-execution.md) - Using built-in backends
- [Custom Tools](tools.md) - Advanced tool patterns
