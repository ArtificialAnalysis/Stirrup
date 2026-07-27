# Changelog

## v0.2.0 (unreleased)

Assistant messages are now block-based (SP-001): an assistant turn is an ordered
sequence of blocks — reasoning, text, tool calls, media — preserving the model's
actual emission order. `blocks` is the only stored content; the channel-era
`content` / `reasoning` / `tool_calls` attributes are read-only projections of it.

### BREAKING

| Operation | v0.2 | Migration |
| --- | --- | --- |
| Read `msg.content` / `.reasoning` / `.tool_calls` | works with a `DeprecationWarning` — read-only projections of blocks | Migrate to `msg.blocks` plus `joined_text`, `reasoning_blocks`, or `tool_call_blocks`. |
| Construct `AssistantMessage(content=…, reasoning=…, tool_calls=…)` | **removed from the typed public API** | Construct `AssistantMessage(blocks=[...])`. Legacy channel-shaped data remains accepted through validation for persisted-history compatibility. |
| Deserialize v0.1 histories (incl. `SubAgentMetadata`, cache files) | works — upgraded at validation, including nullable tool-call/result correlation and the older aggregate cache metadata shape | None. |
| Assign `msg.content = …` / `.tool_calls = …` / `.reasoning = …` | **breaks** — projections have no setters; raises `AttributeError` | Use `msg.with_blocks([…])` or rebuild via the constructor. Grep: `rg "\.(content\|tool_calls\|reasoning)\s*="` |
| External tools reading dumped histories by `content`/`tool_calls` key | **breaks** — dumps emit `blocks` only | Read `blocks` (kind-discriminated), or re-validate through `AssistantMessage` and use the projections. |
| Dumped `ToolCall` payloads | wire change — now carry a `kind: "tool_call"` discriminator key (v0.1 dumps without it still validate; never sent on provider wire formats) | Ignore or read the new key. |
| v0.2 dumps (incl. cache files) read by v0.1 | **breaks** | Upgrade readers to ≥ 0.2. |
| Provide both `blocks` and non-empty channel kwargs | **raises `ValueError`** (new guard) | Pass one representation. |
| `Reasoning` class used as a standalone type | deprecated — survives only as the `reasoning` projection carrier | Match on `ReasoningBlock` / `SignedReasoningBlock` / `RedactedReasoningBlock` / `ReasoningRefBlock` / `EncryptedReasoningBlock`. |
| `from stirrup import SummaryMessage` | **removed** from the top-level namespace (symmetry with `TurnWarningMessage`) | Import from `stirrup.core.models`. |
| Validate a raw user-message dict without a `kind` key through `ChatMessage` | **breaks** — user-role messages now discriminate on `kind`, and the discriminator must be present | Include `"kind": "user"` (every dump since the field was introduced already carries it), or construct `UserMessage(...)` directly. |
| Assistant `metadata` sent on the wire by OpenAI-compatible replay | **removed** — metadata is integrator/user state, never transmitted | None (was undocumented leakage). |
| OpenAI Responses replay of tool-call turns | wire change — when prior turns are replayed (stateless mode, or history predating the last stored response), items replay in true emission order (message/function_call/reasoning interleaved) instead of message-then-all-calls. In default stateful mode, turns at or before the last stored response are not replayed at all (see `previous_response_id` continuation under Added). | Intentional fidelity fix. Channel-constructed messages still replay in the old order. |
| Unknown OpenAI Responses output item / message content types | **raise `NotImplementedError`** — v0.1 silently skipped them. Affects provider built-in tools enabled via kwargs (e.g. `web_search_call` items). Refusals are handled: their text surfaces as answer text. | Don't enable provider built-in tools on this client, or extend `_parse_response_output` with explicit passback semantics. |
| `OpenResponsesClient` kwargs containing client-owned request keys (`background`, `conversation`, `input`, `instructions`, `max_output_tokens`, `model`, `previous_response_id`, `store`, or `stream`), or colliding with dedicated tool/reasoning configuration | **raises `ValueError`** — these keys otherwise bypass invariants or are silently overwritten | Use the client's dedicated arguments; use `encrypted_reasoning=True` for stateless / ZDR operation. Provider-native `tools` / `tool_choice` and `reasoning` kwargs remain valid when no dedicated configuration conflicts. |
| OpenAI Responses tool results with non-string content | **raise `NotImplementedError`** instead of stringifying and corrupting multimodal content | Convert to a provider-supported string representation before constructing `ToolMessage`. |
| LiteLLM replay of signed thinking | wire change — one `thinking_blocks` entry per signed block (was: merged single entry, first signature only) | Intentional fidelity fix for multi-block signed thinking. |

### Added

- Assistant block types, discriminated on `kind`: `TextBlock`, `ReasoningBlock`
  (in-band), `SignedReasoningBlock` (opaque signature passback),
  `RedactedReasoningBlock` (opaque withheld-reasoning payload), `ReasoningRefBlock`
  (provider-side reference), `EncryptedReasoningBlock` (encrypted reasoning payload
  for stateless / zero-data-retention passback, echoed verbatim in position),
  `OpaqueBlock` (provider-native block carried uninterpreted, for marker/control
  blocks that must survive passback), plus `ToolCall` and the media blocks as
  union members (`AssistantBlock`).
- Accessors `joined_text`, `final_text`, `tool_call_blocks`, `reasoning_blocks`,
  and block replacement via `AssistantMessage.with_blocks`.
- `SummaryMessage.replaced_ids`: ids of the assistant messages a summary replaced.
- Documented integration contract: stable `AssistantMessage.id`, metadata opacity,
  `generate` may return an `AssistantMessage` subclass and the framework preserves it.
- OpenAI Responses stores continuation state once on
  `AssistantMessage.provider_response_id`; stored reasoning items contribute
  readable `ReasoningBlock` content without creating redundant item-reference
  blocks. Stateless reasoning uses `EncryptedReasoningBlock`, preserving its
  content and summary fields separately for exact replay.
- `OpenResponsesClient(encrypted_reasoning=True)`: stateless mode — sends
  `store: false` + `include: ["reasoning.encrypted_content"]` and carries
  reasoning state client-side, for zero-data-retention setups.
- `AssistantMessage.provider_response_id` (new serialized field, `None` by
  default): provider-attached continuation state. In default (stateful) mode
  the OpenAI Responses client records the response id on each turn and
  continues via `previous_response_id`, sending only messages after the latest
  stored response instead of replaying full history; instructions are still
  re-derived from the complete local history. These ids are scoped to the
  provider/project and may expire; a definitive not-found response retries once
  with full local history when it is losslessly replayable, and otherwise raises
  a targeted continuation error. `with_blocks` clears the field — the handle is
  bound to the exact emitted content, so an edited copy must replay in full.
- LiteLLM client captures multiple `thinking_blocks` per turn (previously raised
  `ValueError`) including `redacted_thinking`.

### Fixed

- Responses client no longer joins all message items with `"\n"` or keeps only the
  last reasoning item — ordering and multiplicity survive capture and replay.
- `SummaryMessage` / `TurnWarningMessage` now rehydrate as their own types through
  the `ChatMessage` union (nested `kind` discriminator for user-role messages) —
  previously a dumped `SubAgentMetadata` history containing one failed validation
  on reload.
- New fail-loud guards replacing silent data loss: a Responses reasoning item with
  `encrypted_content` but no id raises (the id is the passback handle); reasoning
  summary entries without text raise; a v0.1 assistant `content` payload that is
  neither string nor list fails validation instead of being dropped; agent
  summarization raises if the summarizer returns no text instead of replacing
  history with an empty summary.
- v0.1 cache keys are tried as a fallback for losslessly projectable initial
  assistant messages, then migrated to the v0.2 key on the next save. Uniform
  legacy content encodings are exhaustive; mixed scalar/list encodings are
  bounded to 256 combinations per historical schema to avoid exponential work.

## v0.1.12 (2026-07-08)

[GitHub release](https://github.com/ArtificialAnalysis/Stirrup/releases/tag/v0.1.12)

### What's Changed

#### Fixes

* Return tool errors for invalid web fetch URLs by @MohamedAkbarally in [#61](https://github.com/ArtificialAnalysis/Stirrup/pull/61)

## v0.1.11 (2026-06-16)

[GitHub release](https://github.com/ArtificialAnalysis/Stirrup/releases/tag/v0.1.11)

### What's Changed

#### Fixes

* Retry Brave search rate limits by @MohamedAkbarally in [#58](https://github.com/ArtificialAnalysis/Stirrup/pull/58)

#### Docs

* Explain context overflow recovery by @MohamedAkbarally in [#54](https://github.com/ArtificialAnalysis/Stirrup/pull/54)

#### Packaging

* Bump to v0.1.11 by @MohamedAkbarally in [#59](https://github.com/ArtificialAnalysis/Stirrup/pull/59)

## v0.1.10 (2026-06-09)

[GitHub release](https://github.com/ArtificialAnalysis/Stirrup/releases/tag/v0.1.10)

### What's Changed

#### Fixes

* Recover from context overflow by @MohamedAkbarally in [#51](https://github.com/ArtificialAnalysis/Stirrup/pull/51)

#### Docs

* Update docs to support redesign by @declanjackson in [#52](https://github.com/ArtificialAnalysis/Stirrup/pull/52)

#### Packaging

* Bump to v0.1.10 by @MohamedAkbarally in [#53](https://github.com/ArtificialAnalysis/Stirrup/pull/53)

## v0.1.9 (2026-05-07)

[GitHub release](https://github.com/ArtificialAnalysis/Stirrup/releases/tag/v0.1.9)

### What's Changed

#### New Features

* Allow passthrough kwargs to E2B sandbox creation by @declanjackson in [#42](https://github.com/ArtificialAnalysis/Stirrup/pull/42)
* Add `create_gate` kwarg to `E2BCodeExecToolProvider` for throttling sandbox creation by @BillyDodds in [#48](https://github.com/ArtificialAnalysis/Stirrup/pull/48)
* Allow multiple finish tools by @declanjackson in [#49](https://github.com/ArtificialAnalysis/Stirrup/pull/49)

#### Fixes

* Deep-merge dict metadata during aggregation by @declanjackson in [#41](https://github.com/ArtificialAnalysis/Stirrup/pull/41)
* Avoid empty text parts when encoding OpenAI message content by @declanjackson in [#50](https://github.com/ArtificialAnalysis/Stirrup/pull/50)
* Serialize Docker image prepare to avoid concurrent-pull race by @BillyDodds in [#47](https://github.com/ArtificialAnalysis/Stirrup/pull/47)
* Fix cache handling by @MohamedAkbarally in [#36](https://github.com/ArtificialAnalysis/Stirrup/pull/36)
* Prevent zombie process leak on `code_exec` timeout by @BillyDodds in [#45](https://github.com/ArtificialAnalysis/Stirrup/pull/45)

### Breaking changes

#### `full_msg_history` no longer appends per turn

`session.run()` returns `full_msg_history: list[list[ChatMessage]]`. As of this
release, a new group is **only** appended on context summarization (plus one final
group at session end) — not once per turn. `len(full_msg_history)` no longer
reflects the agent's turn count.

To count turns, count `AssistantMessage` instances across the flattened history:

```python
from itertools import chain
from stirrup.core.models import AssistantMessage

num_turns = sum(
    1 for m in chain.from_iterable(full_msg_history) if isinstance(m, AssistantMessage)
)
```

### New Contributors

* @BillyDodds made their first contribution in [#48](https://github.com/ArtificialAnalysis/Stirrup/pull/48)

## v0.1.8 (2026-03-10)

[GitHub release](https://github.com/ArtificialAnalysis/Stirrup/releases/tag/v0.1.8)

### What's Changed

#### New Features

* Add Slack integration by @declanjackson in [#28](https://github.com/ArtificialAnalysis/Stirrup/pull/28)
* Only include latest summarization and acknowledgement by @declanjackson in [#31](https://github.com/ArtificialAnalysis/Stirrup/pull/31)

#### Fixes

* Use text lexer for tool result rendering by @declanjackson in [#29](https://github.com/ArtificialAnalysis/Stirrup/pull/29)
* Add `AgentSession` to enforce session creation for `ToolProvider`s by @declanjackson in [#32](https://github.com/ArtificialAnalysis/Stirrup/pull/32)

## v0.1.7 (2026-02-13)

[GitHub release](https://github.com/ArtificialAnalysis/Stirrup/releases/tag/v0.1.7)

### What's Changed

#### New Features

* Prevent successive assistant messages in agent loop by @MohamedAkbarally in [#27](https://github.com/ArtificialAnalysis/Stirrup/pull/27)
* Add speed metric by @declanjackson in [#26](https://github.com/ArtificialAnalysis/Stirrup/pull/26)

## v0.1.6 (2026-02-03)

[GitHub release](https://github.com/ArtificialAnalysis/Stirrup/releases/tag/v0.1.6)

### What's Changed

#### New Features

* Allow subagents to share parent exec env by @declanjackson in [#23](https://github.com/ArtificialAnalysis/Stirrup/pull/23)

#### Fixes

* Fix permissions by @steremma in [#21](https://github.com/ArtificialAnalysis/Stirrup/pull/21)
* Allow absolute host paths in Docker execution environment by @declanjackson in [#24](https://github.com/ArtificialAnalysis/Stirrup/pull/24)

#### Packaging

* Add release script and release workflow by @declanjackson in [#25](https://github.com/ArtificialAnalysis/Stirrup/pull/25)

### New Contributors

* @steremma made their first contribution in [#21](https://github.com/ArtificialAnalysis/Stirrup/pull/21)

## v0.1.5 (2026-01-20)

[GitHub release](https://github.com/ArtificialAnalysis/Stirrup/releases/tag/v0.1.5)

Quick bug-fix release following v0.1.4.

### What's Changed

#### Fixes

* Fix cross execution environment file transfer by @declanjackson in [#19](https://github.com/ArtificialAnalysis/Stirrup/pull/19)

## v0.1.4 (2026-01-19)

[GitHub release](https://github.com/ArtificialAnalysis/Stirrup/releases/tag/v0.1.4)

### What's Changed

#### New Features

* Add browser use tool (browser-use is an optional dependency) by @MohamedAkbarally in [#16](https://github.com/ArtificialAnalysis/Stirrup/pull/16)
* Add OpenAI Responses API client by @declanjackson in [#15](https://github.com/ArtificialAnalysis/Stirrup/pull/15)
* Add option to disable caching of conversation state when an agent run is interrupted by @declanjackson in [#17](https://github.com/ArtificialAnalysis/Stirrup/pull/17)

#### Fixes

* Fix validation for output file paths by @declanjackson in [#14](https://github.com/ArtificialAnalysis/Stirrup/pull/14)
* Fix skills not being passed to sub-agents by @declanjackson in [#13](https://github.com/ArtificialAnalysis/Stirrup/pull/13)
* Update tool message truncation to preserve both start and end content by @declanjackson in [#12](https://github.com/ArtificialAnalysis/Stirrup/pull/12)

## v0.1.3 (2026-01-09)

Tag only — no GitHub release. Summarized from commit history.

### What's Changed

#### New Features

* Support Gemini function-calling thought signatures
* Expose API key in LiteLLM client
* Add success flag to all default tools

#### Fixes

* Fix inconsistent tool param type definitions; ensure params is never `None`
* Use `FINISH_TOOL_NAME` constant instead of the literal `'finish'`

## v0.1.2 (2025-12-19)

Tag only — no GitHub release.

### What's Changed

#### New Features

* Add skills by @declanjackson in [#6](https://github.com/ArtificialAnalysis/Stirrup/pull/6)
* Add `.env.example` by @declanjackson in [#3](https://github.com/ArtificialAnalysis/Stirrup/pull/3)
* Add issue templates by @declanjackson in [#4](https://github.com/ArtificialAnalysis/Stirrup/pull/4)

#### Docs

* Update full customization docs by @declanjackson in [#2](https://github.com/ArtificialAnalysis/Stirrup/pull/2)
* Refine mkdocs styling and add branding assets by @keatingw in [#1](https://github.com/ArtificialAnalysis/Stirrup/pull/1)

### New Contributors

* @keatingw made their first contribution in [#1](https://github.com/ArtificialAnalysis/Stirrup/pull/1)

## v0.1.1 (2025-12-10)

Tag only — no GitHub release.

Initial public release of Stirrup: the lightweight foundation for building agents.
