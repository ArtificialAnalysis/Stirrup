# Caching and Resumption

Stirrup automatically caches agent state on interruptions, allowing you to resume long-running tasks.

## Enabling Resume

Pass `resume=True` to `session()`:

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.tools import DEFAULT_TOOLS

client = ChatCompletionsClient(model="gpt-5")
agent = Agent(client=client, name="researcher", tools=DEFAULT_TOOLS, max_turns=50)

async with agent.session(output_dir="./output", resume=True) as session:
    await session.run("Analyze all datasets in the data folder")
```

## How It Works

1. **On interruption** (Ctrl+C, error, or max turns): Stirrup saves conversation state and execution environment files to `~/.cache/stirrup/<task_hash>/`

2. **On next run with `resume=True`**: If a cache exists for the same prompt, the agent restores state and continues from the last turn

3. **On successful completion**: The cache is automatically cleared (configurable via `clear_on_success`)

```
# First run (interrupted at turn 15)
$ python my_agent.py
^C
Cached state for task abc123...

# Second run (resumes from turn 15)
$ python my_agent.py
Resuming from cached state at turn 15
```

## What Gets Cached

- Conversation messages and history
- Current turn number
- Tool metadata
- All files in the execution environment

## Preserving Caches on Success

By default, caches are cleared on successful completion. To preserve them for inspection or debugging:

```python
async with agent.session(
    resume=True,
    clear_cache_on_success=False,  # Keep cache after success
) as session:
    await session.run("Analyze the data")
```

## Managing Caches

```python
from stirrup.core.cache import CacheManager

cache_manager = CacheManager()

# List all caches
for task_hash in cache_manager.list_caches():
    info = cache_manager.get_cache_info(task_hash)
    print(f"{task_hash}: turn {info['turn']}")

# Clear a specific cache
cache_manager.clear_cache("abc123def456")
```

## Notes

- Cache key is computed from the initial prompt—same prompt = same cache
- Caches are stored locally in `~/.cache/stirrup/`
- Caches are automatically cleared on successful completion (by default)
