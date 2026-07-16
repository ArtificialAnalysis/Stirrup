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

## Output Paths

`save_output_files()` preserves safe relative paths below the requested output
directory. For example, `left/report.txt` and `right/report.txt` remain distinct
instead of both becoming `report.txt`. Traversal and destination collisions are
reported in `SaveOutputFilesResult.failed`; within a single call, a file already
written by an earlier requested output is never silently overwritten by a later
one (the first requested output wins). Absolute source paths are accepted only
when they are beneath a backend-declared root; unmatched absolute paths are
rejected rather than flattened to a basename. Source paths use POSIX semantics,
since execution environments are Linux. Host output collection supports macOS
and Linux; Windows hosts are not supported.

All built-in backends copy outputs and leave sources in the execution
environment. Local and Docker capture requested sources before writing any
destination and replace completed destination files atomically. In each
`SavedFile`, `output_path` is a concrete `Path` for a local destination and a
`PurePosixPath` for a path inside another execution environment.

Custom backends with absolute sandbox paths should override
`output_source_roots()` to return `OutputSourceRoot` entries so paths beneath a
known sandbox root can be preserved relative to that root. When a root is a
directory on the host filesystem, set its `host_path` to enable canonical
(symlink-aware) matching — for example accepting `/tmp/...` spellings of a
`/private/tmp/...` root on macOS. Roots must be absolute paths.

If backend metadata exposes symlink targets, override `resolve_output_source()`
to inspect every source path component, ensure every resolved target stays
inside an execution root before the output is read, and return the resolved
path to read.

Backends that support cross-environment transfers should override
`output_destination_identity()` when they can resolve destination symlinks.
The method must reject destinations outside `output_root` and return the
canonical identity used for duplicate detection. Hard-link and concurrent
filesystem-mutation detection are intentionally outside this contract.

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

The base class handles parsing and validation. Call
`self._prepare_command(cmd)` at the top of `run_command`; it returns
`(argv, rejection)`:

- If `rejection` is set, return it.
- If `argv` is set, execute it directly, without a shell.
- If both are `None` (no allowlist configured), run `cmd` through a shell as
  usual.

See `LocalCodeExecToolProvider.run_command` for an example.

## Next Steps

- [Code Execution Guide](../guides/code-execution.md) - Using built-in backends
- [Custom Tools](tools.md) - Advanced tool patterns
