# View Image Tool Provider

The `ViewImageToolProvider` allows agents to view images from their execution environment.

## ViewImageToolProvider

::: stirrup.tools.view_image.ViewImageToolProvider
    options:
      show_source: true
      members:
        - __init__
        - __aenter__
        - __aexit__

## Usage

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
        ViewImageToolProvider(),  # Auto-detects exec environment
    ],
)

async with agent.session() as session:
    await session.run("Create a chart with matplotlib and view it")
```

## How It Works

1. The `ViewImageToolProvider` is initialized with an optional `exec_env` parameter
2. If not provided, it automatically detects the execution environment from the session
3. The `view_image` tool reads image files from the execution environment
4. Images are returned as `ImageContentBlock` objects for the model to see
