# Caching and Resumption

Stirrup caches accepted agent progress whenever a root session ends, including a
successful one. Pass `resume=True` to restore compatible progress:

```python
async with agent.session(
    output_dir="./output",
    resume=True,
    clear_cache_on_success=True,
) as session:
    await session.run("Analyze the datasets")
```

Caches live under `~/.cache/stirrup/<task_hash>/`. A successful cache-owning
session writes a complete generation, including a copy of the managed working
root, and then clears it; a provider-free direct run clears its cache too. Set
`clear_cache_on_success=False` to retain the generation for inspection.

## Run identity

A cache is selected by a versioned identity containing:

- the initial prompt or messages;
- agent name, model slug, and complete system prompt;
- every model-visible tool's category, name, description, and parameter schema;
- uploaded input and skill destination paths, with SHA-256 digests of the bytes
  actually present in the execution environment.

Enumeration order does not affect identity. Changing any identified behavior or
uploaded content starts a fresh run.

Two exclusions are accepted rather than accidental. Endpoint identity is not part
of the cache key. Neither is a nested sub-agent's own configuration: a sub-agent
contributes only the name, description, and fixed `SubAgentParams` schema of the
tool it is exposed as, so changing its model slug, system prompt, or tool set and
then resuming keeps accepted results produced by the previous configuration.
Clear the cache explicitly after reconfiguring a sub-agent.

Because every model-visible tool contributes to identity, duplicate tool names
are now rejected when an `Agent` is constructed, where they previously resolved
silently to the last tool registered under a repeated name.

The identity format and on-disk layout have independent versions. Legacy,
unversioned, or incompatible state is ignored rather than restored.

## Accepted progress

Cached messages are the canonical resume state. Stirrup validates assistant and
tool-result pairing and derives completed, pending, skipped, and successful
finish calls from that sequence. Ambiguous or malformed sequences are treated as
unavailable. A partial ordered tool batch resumes only calls without accepted
results.

Each accepted tool result is checkpointed before logger callbacks. Text-only
image adaptation checkpoints the placeholder tool result and image-bearing user
message together. A successful finish is checkpointed before progress callbacks,
output export, logger shutdown, or provider teardown. Consequently, resumption
does not replay accepted side effects after one of those later phases fails.

Checkpoints are held in memory and written to disk once, as the root session
exits. Only endings that unwind through session exit are therefore recoverable:
exceptions, turn limits, and SIGINT. A process that dies without unwinding -
SIGKILL, SIGTERM, an OOM kill, or `docker stop` - loses every accepted tool
result even with `cache_on_interrupt=True`, so a long run killed by a rolling
deploy replays from turn 0.

Tool metadata is stored as one flat append-only mapping from tool name to
accepted values. Stirrup-owned metadata retains its type and `Addable` behavior,
while plain mappings remain mappings even when their keys resemble Stirrup's
typed metadata envelope. Application Pydantic models are stored and restored as
plain dictionaries, so a resumed run mixes dictionaries for cached results with
live models for results produced after the resume. Metadata remains accepted when
older conversation context is summarized.

Turn and progress counts are derived from accepted history rather than persisted
as parallel counters. On context overflow, Stirrup keeps the latest accepted
turn and attempts to summarize an older complete-turn prefix. If no safe prefix
fits, accepted history remains checkpointed and the context error is raised.

## State and filesystem generations

State and managed execution files are written into one unique, initially
unselected generation directory. After both are complete, an atomic replacement
of `current.json` selects that generation. All tasks under one cache root share
one process/thread lock and one stable cross-process `.cache.lock` file.

A failed state write, file copy, or pointer replacement leaves the previously
selected state/files generation loadable. Failed and unselected generations are
cleaned when possible. `current.json` is the only atomic selector; sudden power
or storage loss can leave an unselected generation or invalidate the newest
pointer, in which case that cache is treated as unavailable.

For local and Docker providers, restore exactly replaces the provider's managed
`temp_dir`, including removal of fresh uploads absent from the selected snapshot.
Files outside that managed root are untouched. Providers without a managed local
root, such as remote-only sandboxes, start fresh instead of attempting a partial
filesystem restore.

Cache directories and control files are private to the creating user. Symlinked
cache, task, or generation directories are rejected, and cleanup never follows a
symlink. Malformed pointers and state are treated as unavailable; a cache that
exists but cannot be loaded is reported as unusable rather than absent, because
starting fresh re-executes every previously accepted tool call. Substantive
filesystem failures such as permission errors are propagated.

## Managing caches

```python
from stirrup.core.cache import CacheManager

cache_manager = CacheManager()
for task_hash in cache_manager.list_caches():
    print(cache_manager.get_cache_info(task_hash))

cache_manager.clear_cache("<task_hash>")
```

Within one process, concurrent cache-enabled or resuming root runs with the
same complete task identity are rejected explicitly. Different identities can
run concurrently. Set `cache_on_interrupt=False` only when identical runs must
overlap and resumable side-effect guarantees are not required; a cache-disabled
run neither writes nor clears another run's task cache.
