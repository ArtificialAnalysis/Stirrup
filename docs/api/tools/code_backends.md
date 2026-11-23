# Code Execution Backends

The `stirrup.tools.code_backends` module provides code execution backends.

## CodeExecToolProvider (Base Class)

::: stirrup.tools.code_backends.CodeExecToolProvider
    options:
      show_source: true
      members:
        - __init__
        - run_command
        - read_file_bytes
        - write_file_bytes
        - save_output_files
        - upload_files
        - get_code_exec_tool
        - get_view_image_tool

## LocalCodeExecToolProvider

Executes code in an isolated temporary directory on the host machine.

::: stirrup.tools.code_backends.LocalCodeExecToolProvider
    options:
      show_source: true

## DockerCodeExecToolProvider

Executes code in a Docker container.

!!! note
    Requires `pip install stirrup[docker]` (or: `uv add stirrup[docker]`)

```python
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

provider = DockerCodeExecToolProvider.from_image("python:3.12-slim")
```

## E2BCodeExecToolProvider

Executes code in an E2B cloud sandbox.

!!! note
    Requires `pip install stirrup[e2b]` (or: `uv add stirrup[e2b]`) and `E2B_API_KEY` environment variable.

```python
from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider

provider = E2BCodeExecToolProvider()
```

## DaytonaCodeExecToolProvider

Executes code in a Daytona cloud sandbox.

!!! note
    Requires `pip install stirrup[daytona]` (or: `uv add stirrup[daytona]`) and `DAYTONA_API_KEY` environment variable.

```python
from stirrup.tools.code_backends.daytona import DaytonaCodeExecToolProvider

provider = DaytonaCodeExecToolProvider()
```

::: stirrup.tools.code_backends.daytona.DaytonaCodeExecToolProvider
    options:
      show_source: true

## Data Types

::: stirrup.tools.code_backends.CodeExecutionParams

::: stirrup.tools.code_backends.CommandResult

::: stirrup.tools.code_backends.SavedFile

::: stirrup.tools.code_backends.SaveOutputFilesResult

::: stirrup.tools.code_backends.UploadedFile

::: stirrup.tools.code_backends.UploadFilesResult
