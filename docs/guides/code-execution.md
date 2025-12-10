# Code Execution

Stirrup provides multiple backends for executing code in isolated environments.

## Overview

All code execution backends implement `CodeExecToolProvider`, which provides:

- A `code_exec` tool for running shell commands
- File upload/download capabilities
- Isolated execution environment

## Available Backends

| Backend | Isolation | Use Case |
|---------|-----------|----------|
| `LocalCodeExecToolProvider` | Temp directory | Development, trusted code |
| `DockerCodeExecToolProvider` | Container | Production, semi-trusted code |
| `E2BCodeExecToolProvider` | Cloud sandbox | Production, untrusted code |

## LocalCodeExecToolProvider

Executes code in an isolated temporary directory on the host machine.

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider

client = ChatCompletionsClient(model="gpt-5")
agent = Agent(
    client=client,
    name="local_coder",
    tools=[LocalCodeExecToolProvider()],
)

async with agent.session(output_dir="./output") as session:
    await session.run("Write and run a Python hello world script")
```

### Configuration

```python
from stirrup.tools import LocalCodeExecToolProvider

provider = LocalCodeExecToolProvider(
    allowed_commands=None,      # Regex patterns for allowed commands (None = all)
    temp_base_dir=None,         # Base directory for temp folder (default: system temp)
)
```

### Security Considerations

- Code runs with your user permissions
- Absolute paths outside temp directory are blocked
- Use `allowed_commands` to restrict what can be executed

```python
from stirrup.tools import LocalCodeExecToolProvider

# Only allow Python and pip
provider = LocalCodeExecToolProvider(
    allowed_commands=["python.*", "pip.*", "uv.*"],
)
```

## DockerCodeExecToolProvider

Executes code in a Docker container for better isolation.

!!! note
    Requires `uv pip install stirrup[docker]` and Docker daemon running.

### From Image

```python
from stirrup import Agent
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

provider = DockerCodeExecToolProvider.from_image(
    "python:3.12-slim",
    env_vars=["OPENAI_API_KEY"],  # Forward these env vars
)

client = ChatCompletionsClient(model="gpt-5")
agent = Agent(
    client=client,
    name="docker_coder",
    tools=[provider],
)
```

### From Dockerfile

```python
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

provider = DockerCodeExecToolProvider.from_dockerfile(
    dockerfile_path="./Dockerfile",
    build_args={"PYTHON_VERSION": "3.12"},
)
```

### Configuration Options

```python
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider

provider = DockerCodeExecToolProvider.from_image(
    "python:3.12-slim",
    env_vars=["API_KEY", "SECRET"],  # Environment variables to forward
)
```

## E2BCodeExecToolProvider

Executes code in E2B cloud sandboxes for maximum isolation.

!!! note
    Requires `uv pip install stirrup[e2b]` and `E2B_API_KEY` environment variable.

```python
from stirrup import Agent
from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider

provider = E2BCodeExecToolProvider()

client = ChatCompletionsClient(model="gpt-5")
agent = Agent(
    client=client,
    name="e2b_coder",
    tools=[provider],
    max_turns=20,
)

async with agent.session(
    input_files="data/*.csv",
    output_dir="./results",
) as session:
    await session.run("Analyze the CSV files and create a report")
```

### With Template

```python
from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider

provider = E2BCodeExecToolProvider(
    template="custom-python-template",  # Your E2B template ID
)
```

## File Operations

### Uploading Files

Use `input_files` in session to upload files to the execution environment:

```python
async with agent.session(
    input_files="data.csv",           # Single file
    # input_files=["a.csv", "b.csv"], # Multiple files
    # input_files="data/*.csv",       # Glob pattern
    # input_files="./data/",          # Directory (recursive)
) as session:
    await session.run("Process the uploaded files")
```

### Saving Output Files

When the agent calls the finish tool with file paths, they're saved to `output_dir`:

```python
async with agent.session(output_dir="./output") as session:
    finish_params, _, _ = await session.run(
        "Create a report and save it as report.pdf"
    )
    # Files in finish_params.paths are saved to ./output/
    print(f"Saved: {finish_params.paths}")
```

## View Image Tool

Use `ViewImageToolProvider` to let the agent view images from the execution environment:

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider, ViewImageToolProvider

client = ChatCompletionsClient(model="gpt-5")
agent = Agent(
    client=client,
    name="image_viewer",
    tools=[
        LocalCodeExecToolProvider(),
        ViewImageToolProvider(),  # Auto-detects exec env
    ],
)

async with agent.session() as session:
    await session.run("Create a matplotlib chart and view it")
```

## Next Steps

- [Sub-Agents](sub-agents.md) - File transfer between agents
- [Extending Backends](../extending/code_backends.md) - Custom execution backends
