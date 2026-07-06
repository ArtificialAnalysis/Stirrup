# Changelog

## v0.2.0

Assistant messages are now block-based (SP-001): an assistant turn is an ordered
sequence of blocks — reasoning, text, tool calls, media — preserving the model's
actual emission order. `blocks` is the only stored content; the channel-era
`content` / `reasoning` / `tool_calls` attributes are read-only projections of it.

### BREAKING

| Operation | v0.2 | Migration |
| --- | --- | --- |
| Read `msg.content` / `.reasoning` / `.tool_calls` | works — read-only projections of blocks | None. Prefer `msg.blocks` / `final_text(msg.blocks)` in new code. |
| Construct `AssistantMessage(content=…, reasoning=…, tool_calls=…)` | works — blocks synthesized in channel order (reasoning → text → tool calls); permanent path | None. Custom `LLMClient`s keep working unmodified. |
| Deserialize v0.1 histories (incl. `SubAgentMetadata`, cache files) | works — upgraded at validation | None. |
| Assign `msg.content = …` / `.tool_calls = …` / `.reasoning = …` | **breaks** — projections have no setters; raises `AttributeError` | `msg = msg.with_text("…")`, `msg.with_blocks([…])`, or rebuild via the constructor. Grep: `rg "\.(content\|tool_calls\|reasoning)\s*="` |
| External tools reading dumped histories by `content`/`tool_calls` key | **breaks** — dumps emit `blocks` only | Read `blocks` (kind-discriminated), or re-validate through `AssistantMessage` and use the projections. |
| Dumped `ToolCall` payloads | wire change — now carry a `kind: "tool_call"` discriminator key (v0.1 dumps without it still validate; never sent on provider wire formats) | Ignore or read the new key. |
| v0.2 dumps (incl. cache files) read by v0.1 | **breaks** | Upgrade readers to ≥ 0.2. |
| Provide both `blocks` and non-empty channel kwargs | **raises `ValueError`** (new guard) | Pass one representation. |
| `Reasoning` class used as a standalone type | deprecated — survives only as the `reasoning` projection carrier | Match on `ReasoningBlock` / `SignedReasoningBlock` / `RedactedReasoningBlock` / `ReasoningRefBlock`. |
| `from stirrup import SummaryMessage` | **removed** from the top-level namespace (symmetry with `TurnWarningMessage`) | Import from `stirrup.core.models`. |
| Validate a raw user-message dict without a `kind` key through `ChatMessage` | **breaks** — user-role messages now discriminate on `kind`, and the discriminator must be present | Include `"kind": "user"` (every dump since the field was introduced already carries it), or construct `UserMessage(...)` directly. |
| Assistant `metadata` sent on the wire by OpenAI-compatible replay | **removed** — metadata is integrator/user state, never transmitted | None (was undocumented leakage). |
| OpenAI Responses replay of tool-call turns | wire change — items now replay in true emission order (message/function_call/reasoning interleaved) instead of message-then-all-calls | Intentional fidelity fix. Channel-constructed messages still replay in the old order. |
| LiteLLM replay of signed thinking | wire change — one `thinking_blocks` entry per signed block (was: merged single entry, first signature only) | Intentional fidelity fix for multi-block signed thinking. |

### Added

- Assistant block types, discriminated on `kind`: `TextBlock`, `ReasoningBlock`
  (in-band), `SignedReasoningBlock` (opaque signature passback),
  `RedactedReasoningBlock` (opaque withheld-reasoning payload), `ReasoningRefBlock`
  (provider-side reference, with optional `encrypted_content` for ZDR),
  `OpaqueBlock` (provider-native block carried uninterpreted, for marker/control
  blocks that must survive passback), plus `ToolCall` and the media blocks as
  union members (`AssistantBlock`).
- Accessors `joined_text`, `final_text`, `tool_call_blocks`, `reasoning_blocks`,
  and message mutators `AssistantMessage.with_text` / `.with_blocks`.
- `SummaryMessage.replaced_ids`: ids of the assistant messages a summary replaced.
- Documented integration contract: stable `AssistantMessage.id`, metadata opacity,
  `generate` may return an `AssistantMessage` subclass and the framework preserves it.
- OpenAI Responses client now captures reasoning items as `ReasoningRefBlock`
  (id + summary + `encrypted_content` when requested) and re-emits them on replay;
  previously reasoning was reduced to summary text and dropped on replay.
- LiteLLM client captures multiple `thinking_blocks` per turn (previously raised
  `ValueError`) including `redacted_thinking`.

### Fixed

- Responses client no longer joins all message items with `"\n"` or keeps only the
  last reasoning item — ordering and multiplicity survive capture and replay.
- `SummaryMessage` / `TurnWarningMessage` now rehydrate as their own types through
  the `ChatMessage` union (nested `kind` discriminator for user-role messages) —
  previously a dumped `SubAgentMetadata` history containing one failed validation
  on reload, losing `replaced_ids` lineage.
