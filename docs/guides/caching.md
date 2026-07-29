# Caching and Resumption

Stirrup automatically caches agent state on interruptions, so a long run that is stopped by Ctrl+C, an error, or `max_turns` can be picked up again instead of restarted.

A cache is resumed **only when every model-visible input still matches exactly**. Change the model, edit a tool description, change the `system_prompt` you pass to `Agent()`, or edit an input file, and the next run starts fresh rather than resuming. That is deliberate — see [Why resume is brittle](#why-resume-is-brittle).

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

2. **On next run with `resume=True`**: If a cache exists whose identity matches this run exactly, the agent restores state and continues from the last accepted turn

3. **On successful completion**: The cache is automatically cleared (configurable via `clear_on_success`)

```
# First run (interrupted at turn 15)
$ python my_agent.py
^C
Cached state for task 9f3c...

# Second run (resumes from turn 15)
$ python my_agent.py
Resuming from cached state at turn 15
```

## What Gets Cached

- Conversation messages and history
- Tool metadata keyed by accepted assistant turn
- All files in the execution environment

Turn progress is derived from the restored message history rather than stored separately. This keeps cached runs aligned with context-overflow recovery, where an unwound turn is removed from both history and metadata.

## Task Identity

The cache key (`task_hash`) is a digest of everything that shapes the transcript:

- The identity version — a constant bumped whenever the payload or the base system prompt template changes, which invalidates every existing cache at once
- The initial prompt or message list, tagged by kind so a prompt cannot collide with the serialization of a message list
- The agent name
- The model slug
- The `system_prompt` passed to `Agent()`
- Every tool the model can see: name, description, parameter schema, and whether it is a finish tool — including tools supplied by an MCP server
- The **content** of every file uploaded into the execution environment, paired with its destination path
- The **content** of every file in the uploaded skills tree, plus each skill's name and description

Content, not filenames: running a task with `data.csv`, interrupting it, editing `data.csv`, and resuming would otherwise continue a transcript whose conclusions were computed against the old bytes — a confident wrong answer with no error.

### Deliberately excluded

- **`max_turns`.** Raising it and resuming is the main reason a max-turns cache exists, so including it would refuse that resume every time. The consequence: after resuming with a raised limit, the restored system message still states the original step budget.
- **Sampling parameters** (temperature, top_p, client `max_tokens`), **API endpoint, base URL and credentials.** None of these are shown to the model, and a credential must never become part of a value used as a directory name.
- **Run-control knobs**: `context_summarization_cutoff`, `turns_remaining_warning_threshold`, `block_successive_assistant_messages`, `text_only_tool_responses`, `output_dir`. The messages these inject are already recorded verbatim in the cached transcript, so adjusting them mid-recovery is supported and intended.

### What identity does *not* cover

Identity covers the contract the model is shown and the declared inputs to the task — not the behaviour of the code behind that contract. Edit a custom tool's Python body, change what an MCP server does behind an unchanged description, or change a sub-agent's model, system prompt or tool list, and the identity is unchanged: the resume proceeds against a transcript whose results came from different code.

## The Three Outcomes

With `resume=True` a run reports exactly one of:

| Outcome | Message |
| --- | --- |
| No cache for this identity | `No cache found for task <hash>, starting fresh` |
| A cache exists but is unusable | `Found a cache for task <hash> but it cannot be resumed (<reason>); starting fresh. Tool calls from the interrupted run will be made again.` |
| Cache accepted | `Resuming from cached state at turn N` |

A cache is refused, never partially trusted, when its identity version does not match, its stored identity does not match, its state file is missing or malformed, its recorded files are gone, there is no execution environment to restore files into, or the file restore fails partway. Refusal is reported distinctly from absence because the two mean different things to you: absence is expected, refusal means work you already paid for is being redone.

## Why Resume Is Brittle

Resume is a Ctrl+C recovery aid, not a memoization layer. A resume that silently continues against changed inputs produces a confident wrong answer with no error and no warning — strictly worse than losing the turns and starting again. So identity is matched exactly and a mismatch always starts fresh.

In practice this means: change the model, add, remove, reorder or re-describe a tool, change a tool's parameter schema, change the `system_prompt`, edit an input file, or edit a skill file, and you start over.

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
cache_manager.clear_cache(task_hash)
```

## Limits

- **Identity is backend-specific.** The same task on a local backend and on Docker hashes differently, because the file paths shown to the model differ.
- **On E2B there is no local temp directory**, so `files/` is never written and a resume restores the transcript only.
- **Establishing identity re-reads every uploaded input file and skill file** back out of the execution environment at session start — one round trip per file on a container backend, on every session, including sessions that never interrupt.
- **Run metadata from turns completed before the interruption is restored as plain JSON**, not as the Pydantic models a fresh run returns.
- **A cache torn by a crash during the file copy is refused, not repaired.** `state.json` is the commit point: it is removed before the files are refreshed and written back last, so an interrupted save leaves an uncommitted cache that the next run refuses rather than a state file paired with half-copied files. There is no previous generation to fall back to.
- **Concurrent runs of the identical task can clobber each other's cache.** Cache writes are not transactional and nothing serializes them, within one process or across processes; the last writer wins, and the loser's cache is refused on the next resume rather than silently mixed. Give overlapping runs distinct identities if that matters, or set `cache_on_interrupt=False`.
- **Only depth-0 runs have a cache identity.** Sub-agents never save or resume, and their input files are not fingerprinted.
- **The cache is not a security boundary.** It lives under `~/.cache/stirrup/` with default umask permissions and follows symlinks when copying files. If another local user can write to your cache root, do not use `resume=True`.
- **Caches written by an older stirrup are never resumed** and are not cleaned up either; `~/.cache/stirrup/` may hold orphaned directories until you remove them.
