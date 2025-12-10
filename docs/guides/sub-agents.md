# Sub-Agents

Sub-agents allow you to break complex tasks into specialized agents that work together.

## Overview

Any `Agent` can be converted to a `Tool` using the `.to_tool()` method. This allows a parent agent to delegate tasks to specialized child agents.

## Basic Pattern

```python
import asyncio

from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import WebToolProvider


async def main():
    # Create client for OpenAI (shared across agents)
    client = ChatCompletionsClient(
        base_url="https://api.openai.com/v1",
        model="gpt-5",
    )

    # Create a specialist agent
    researcher = Agent(
        client=client,
        name="researcher",
        tools=[WebToolProvider()],
        system_prompt="You are a research specialist. Find accurate information.",
    )

    # Convert to a tool for the parent
    research_tool = researcher.to_tool(
        description="Research a topic and return findings",
    )

    # Parent agent uses the sub-agent
    orchestrator = Agent(
        client=client,
        name="orchestrator",
        tools=[research_tool],
    )

    async with orchestrator.session() as session:
        await session.run(
            "Research the latest developments in quantum computing"
        )


asyncio.run(main())
```

## The `to_tool()` Method

```python
agent.to_tool(
    description="Description of what this sub-agent does",
    system_prompt=None,  # Optional override for sub-agent's system prompt
)
```

Returns a `Tool[SubAgentParams, SubAgentMetadata]`.

### SubAgentParams

When the parent calls the sub-agent tool, it provides:

```python
class SubAgentParams(BaseModel):
    task: str              # The task for the sub-agent
    input_files: list[str] # Files to pass from parent to sub-agent
```

### SubAgentMetadata

The sub-agent returns metadata about its execution:

```python
class SubAgentMetadata(BaseModel):
    message_history: list[list[ChatMessage]]  # Sub-agent's conversation
    run_metadata: dict[str, list[Any]]        # Tool metadata from sub-agent
```

## File Transfer Between Agents

!!! warning "Critical Requirement"
    If a sub-agent has a code execution environment and produces files, the parent agent **must** also have a `CodeExecToolProvider`.

### Parent Without Code Exec (No File Transfer)

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import WebToolProvider

client = ChatCompletionsClient(
    base_url="https://api.openai.com/v1",
    model="gpt-5",
)

# Sub-agent does web research only (no files produced)
researcher = Agent(
    client=client,
    name="researcher",
    tools=[WebToolProvider()],  # No code exec
)

# Parent doesn't need code exec
parent = Agent(
    client=client,
    name="parent",
    tools=[researcher.to_tool(description="Web researcher")],
)
```

### Parent With Code Exec (File Transfer Enabled)

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider

client = ChatCompletionsClient(
    base_url="https://api.openai.com/v1",
    model="gpt-5",
)

# Sub-agent creates files
coder = Agent(
    client=client,
    name="coder",
    tools=[LocalCodeExecToolProvider()],
)

# Parent MUST have code exec to receive files
parent = Agent(
    client=client,
    name="parent",
    tools=[
        LocalCodeExecToolProvider(),  # Required!
        coder.to_tool(description="Write and test code"),
    ],
)
```

### How File Transfer Works

1. Sub-agent creates files in its execution environment
2. Sub-agent calls finish tool with `paths` listing the files
3. Files are automatically transferred to parent's execution environment
4. Parent can reference these files in subsequent operations

## Multiple Sub-Agents

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider, WebToolProvider

client = ChatCompletionsClient(
    base_url="https://api.openai.com/v1",
    model="gpt-5",
)

# Specialist agents
researcher = Agent(
    client=client,
    name="researcher",
    tools=[WebToolProvider()],
)

coder = Agent(
    client=client,
    name="coder",
    tools=[LocalCodeExecToolProvider()],
)

reviewer = Agent(
    client=client,
    name="reviewer",
    tools=[LocalCodeExecToolProvider()],
)

# Orchestrator with all sub-agents
orchestrator = Agent(
    client=client,
    name="orchestrator",
    tools=[
        LocalCodeExecToolProvider(),  # For file transfer
        researcher.to_tool(description="Research a topic"),
        coder.to_tool(description="Write code based on requirements"),
        reviewer.to_tool(description="Review and test code"),
    ],
)

async with orchestrator.session(output_dir="./output") as session:
    await session.run(
        "Research best practices for Python CLI tools, write one, and review it"
    )
```

## Nested Sub-Agents

Sub-agents can themselves have sub-agents:

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider, WebToolProvider

client = ChatCompletionsClient(
    base_url="https://api.openai.com/v1",
    model="gpt-5",
)

# Level 2: Specialist
data_fetcher = Agent(
    client=client,
    name="data_fetcher",
    tools=[WebToolProvider()],
)

# Level 1: Analyst uses data fetcher
analyst = Agent(
    client=client,
    name="analyst",
    tools=[
        LocalCodeExecToolProvider(),
        data_fetcher.to_tool(description="Fetch data from APIs"),
    ],
)

# Level 0: Orchestrator uses analyst
orchestrator = Agent(
    client=client,
    name="orchestrator",
    tools=[
        LocalCodeExecToolProvider(),
        analyst.to_tool(description="Analyze data and produce reports"),
    ],
)
```

## Accessing Sub-Agent Results

Sub-agent metadata is included in the parent's run metadata:

```python
finish_params, history, metadata = await session.run("Use sub-agent")

from stirrup import aggregate_metadata

aggregated = aggregate_metadata(metadata)

# Access sub-agent metadata
if "researcher" in aggregated:
    sub_meta = aggregated["researcher"]
    print(f"Sub-agent history: {len(sub_meta.message_history)} groups")
```

## Use Cases

### Research → Code → Test Pipeline

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import LocalCodeExecToolProvider

client = ChatCompletionsClient(base_url="https://api.openai.com/v1", model="gpt-5")

# Each agent specializes in one task
researcher = Agent(client=client, name="researcher", ...)
coder = Agent(client=client, name="coder", ...)
tester = Agent(client=client, name="tester", ...)

# Orchestrator coordinates the workflow
orchestrator = Agent(
    client=client,
    name="orchestrator",
    tools=[
        LocalCodeExecToolProvider(),
        researcher.to_tool(description="Research requirements"),
        coder.to_tool(description="Implement the solution"),
        tester.to_tool(description="Write and run tests"),
    ],
)
```

### Parallel Research

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient

client = ChatCompletionsClient(base_url="https://api.openai.com/v1", model="gpt-5")

# Multiple research specialists
web_researcher = Agent(client=client, name="web_researcher", ...)
doc_researcher = Agent(client=client, name="doc_researcher", ...)

# Parent can call them in any order
parent = Agent(
    client=client,
    name="coordinator",
    tools=[
        web_researcher.to_tool(description="Search the web"),
        doc_researcher.to_tool(description="Search documentation"),
    ],
)
```

## Limitations

- Sub-agent runs are synchronous (parent waits for completion)
- All sub-agent messages are returned to parent (may use context)
- File transfer only works with code execution environments

## Next Steps

- [Code Execution](code-execution.md) - Execution backend options
- [MCP Integration](mcp.md) - External tool servers
