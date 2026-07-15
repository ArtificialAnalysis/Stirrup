import base64
import mimetypes
import warnings
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Awaitable, Callable, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import BytesIO
from math import isinf, isnan, sqrt
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Annotated, Any, ClassVar, Literal, Protocol, Self, cast, overload, runtime_checkable
from uuid import uuid4

import filetype
from moviepy import AudioFileClip, VideoFileClip
from moviepy.video.fx import Resize
from PIL import Image
from pydantic import BaseModel, Field, PlainSerializer, PlainValidator, model_validator

from stirrup.constants import RESOLUTION_1MP, RESOLUTION_480P

__all__ = [
    "Addable",
    "AnyReasoningBlock",
    "AssistantBlock",
    "AssistantMessage",
    "AudioContentBlock",
    "BinaryContentBlock",
    "ChatMessage",
    "Content",
    "ContentBlock",
    "EmptyParams",
    "EncryptedReasoningBlock",
    "ImageContentBlock",
    "LLMClient",
    "OpaqueBlock",
    "Reasoning",
    "ReasoningBlock",
    "ReasoningRefBlock",
    "RedactedReasoningBlock",
    "SignedReasoningBlock",
    "SubAgentMetadata",
    "SummaryMessage",
    "SystemMessage",
    "TextBlock",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "ToolProvider",
    "ToolResult",
    "ToolUseCountMetadata",
    "TurnWarningMessage",
    "UserMessage",
    "VideoContentBlock",
    "aggregate_metadata",
    "final_text",
    "joined_text",
    "reasoning_blocks",
    "tool_call_blocks",
]


def _bytes_to_b64(v: bytes) -> str:
    return base64.b64encode(v).decode("ascii")


def _b64_to_bytes(v: bytes | str) -> bytes:
    if isinstance(v, bytes):
        return v
    if isinstance(v, str):
        return base64.b64decode(v.encode("ascii"))
    raise TypeError("Invalid bytes value")


Base64Bytes = Annotated[
    bytes,
    PlainValidator(_b64_to_bytes),
    PlainSerializer(_bytes_to_b64, when_used="json"),
]


def downscale_image(w: int, h: int, max_pixels: int | None = 1_000_000) -> tuple[int, int]:
    """Downscale image dimensions to fit within max pixel count while maintaining aspect ratio.

    Returns even dimensions with minimum 2x2 size.
    """
    s = 1.0 if max_pixels is None or w * h <= max_pixels else sqrt(max_pixels / (w * h))
    nw, nh = int(w * s) // 2 * 2, int(h * s) // 2 * 2
    return max(nw, 2), max(nh, 2)


# Content
class BinaryContentBlock(BaseModel, ABC, frozen=True):
    """Base class for binary content (images, video, audio) with MIME type validation."""

    data: Base64Bytes
    allowed_mime_types: ClassVar[set[str]]

    @property
    def mime_type(self) -> str:
        """MIME type for data based on headers."""
        match: filetype.Type = filetype.guess(self.data)
        if match is None:
            raise ValueError(f"Unsupported file type {self.data!r}")
        return match.mime

    @property
    def extension(self) -> str:
        """File extension for the content (e.g., 'png', 'mp4', 'mp3') without leading dot."""
        _extension = mimetypes.guess_extension(self.mime_type)
        if _extension is None:
            raise ValueError(f"Unsupported mime_type {self.mime_type!r}")
        return _extension[1:]

    @model_validator(mode="after")
    def _check_mime(self) -> Self:
        """Validate MIME type against allowed list and verify content is readable."""
        if self.allowed_mime_types and self.mime_type not in self.allowed_mime_types:
            raise ValueError(f"Unsupported mime_type {self.mime_type!r}; allowed: {self.allowed_mime_types}")
        self._probe()  # light corruption check; no heavy work
        return self

    @abstractmethod
    def _probe(self) -> None:
        """Verify content can be opened and read; subclasses implement format-specific checks."""


class ImageContentBlock(BinaryContentBlock, frozen=True):
    """Image content supporting PNG, JPEG, WebP, PSD formats with automatic downscaling."""

    kind: Literal["image_content_block"] = "image_content_block"
    allowed_mime_types: ClassVar[set[str]] = {
        "image/jpeg",  # JPEG
        "image/png",  # PNG
        "image/webp",  # WebP
        "image/gif",  # GIF
        "image/bmp",  # BMP
        "image/tiff",  # TIFF
        "image/vnd.adobe.photoshop",  # PSD
    }

    def _probe(self) -> None:
        """Verify image data is valid by attempting to open and verify it with PIL."""
        with Image.open(BytesIO(self.data)) as im:
            im.verify()

    def to_base64_url(self, max_pixels: int | None = RESOLUTION_1MP) -> str:
        """Convert image to base64 data URL, optionally resizing to max pixel count."""
        img: Image.Image = Image.open(BytesIO(self.data))
        if max_pixels is not None and img.width * img.height > max_pixels:
            tw, th = downscale_image(img.width, img.height, max_pixels)
            img.thumbnail((tw, th), Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = BytesIO()
        img.save(buf, format="PNG")
        return f"data:image/png;base64,{b64encode(buf.getvalue()).decode()}"


class VideoContentBlock(BinaryContentBlock, frozen=True):
    """MP4 video content with automatic transcoding and resolution downscaling."""

    kind: Literal["video_content_block"] = "video_content_block"
    allowed_mime_types: ClassVar[set[str]] = {
        "video/x-msvideo",  # AVI
        "video/mp4",  # MP4
        "video/quicktime",  # MOV
        "video/x-matroska",  # MKV
        "video/x-ms-wmv",  # WMV
        "video/x-flv",  # FLV
        "video/mpeg",  # MPEG
        "video/webm",  # WebM
        "video/gif",  # GIF (animated)
    }

    def _probe(self) -> None:
        """Verify video data is valid by attempting to open it as a VideoFileClip."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.*")
        with NamedTemporaryFile(suffix=".bin") as f:
            f.write(self.data)
            f.flush()
            clip = VideoFileClip(f.name)
            clip.close()

    def to_base64_url(self, max_pixels: int | None = RESOLUTION_480P, fps: int | None = None) -> str:
        """Transcode to MP4 and return base64 data URL."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.*")
            with NamedTemporaryFile(suffix=".mp4") as fin, NamedTemporaryFile(suffix=".mp4") as fout:
                fin.write(self.data)
                fin.flush()
                clip = VideoFileClip(fin.name)
                tw, th = downscale_image(int(clip.w), int(clip.h), max_pixels)
                clip = clip.with_effects([Resize(new_size=(tw, th))])

                clip.write_videofile(
                    fout.name,
                    codec="libx264",
                    fps=fps,
                    audio=clip.audio is not None,
                    audio_codec="aac",
                    preset="veryfast",
                    logger=None,
                )
                clip.close()
                return f"data:video/mp4;base64,{b64encode(fout.read()).decode()}"


class AudioContentBlock(BinaryContentBlock, frozen=True):
    """Audio content supporting MPEG, WAV, AAC, and other common audio formats."""

    kind: Literal["audio_content_block"] = "audio_content_block"
    allowed_mime_types: ClassVar[set[str]] = {
        "audio/x-aac",
        "audio/flac",
        "audio/mp3",
        "audio/m4a",
        "audio/mpeg",
        "audio/mpga",
        "audio/mp4",
        "audio/ogg",
        "audio/pcm",
        "audio/wav",
        "audio/webm",
        "audio/x-wav",
        "audio/aac",
    }

    def _probe(self) -> None:
        """Verify audio data is valid by attempting to open it as an AudioFileClip."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.*")
        with NamedTemporaryFile(suffix=".bin") as fin:
            fin.write(self.data)
            fin.flush()
            clip = AudioFileClip(fin.name)
            clip.close()

    def to_base64_url(self, bitrate: str = "192k") -> str:
        """Transcode to MP3 and return base64 data URL."""
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="moviepy.*")
            with NamedTemporaryFile(suffix=".bin") as fin, NamedTemporaryFile(suffix=".mp3") as fout:
                fin.write(self.data)
                fin.flush()
                clip = AudioFileClip(fin.name)
                clip.write_audiofile(fout.name, codec="libmp3lame", bitrate=bitrate, logger=None)
                clip.close()
                return f"data:audio/mpeg;base64,{b64encode(fout.read()).decode()}"


type ContentBlock = ImageContentBlock | VideoContentBlock | AudioContentBlock | str
"""Union of all content block types (image, video, audio, or text)."""

type Content = list[ContentBlock] | str
"""Message content: either a plain string or list of mixed content blocks."""


# Metadata Protocol and Aggregation
@runtime_checkable
class Addable(Protocol):
    """Protocol for types that support aggregation via __add__."""

    def __add__(self, other: Self) -> Self: ...


def _merge_dicts(a: dict, b: dict) -> dict:
    """Deep merge two dicts, recursively merging nested dicts and summing numbers."""
    merged = dict(a)
    for key, value in b.items():
        if key in merged:
            existing = merged[key]
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = _merge_dicts(existing, value)
            elif (isinstance(existing, int | float) and isinstance(value, int | float)) or (
                isinstance(existing, list) and isinstance(value, list)
            ):
                merged[key] = existing + value
            else:
                merged[key] = value
        else:
            merged[key] = value
    return merged


def _aggregate_list[T: Addable](metadata_list: list[T]) -> T | None:
    """Aggregate a list of metadata using __add__, with fallback for dicts."""
    if not metadata_list:
        return None
    aggregated: T = metadata_list[0]
    for m in metadata_list[1:]:
        if isinstance(aggregated, dict) and isinstance(m, dict):
            aggregated = _merge_dicts(aggregated, m)  # ty: ignore[invalid-assignment]
        else:
            aggregated = aggregated + m
    return aggregated


def to_json_serializable(value: object) -> object:
    # None and JSON primitives
    if value is None or isinstance(value, str | int | bool):
        return value

    # Floats need special handling for nan/inf
    if isinstance(value, float):
        if isnan(value) or isinf(value):
            raise ValueError(f"Cannot serialize {value} to JSON")
        return value

    # Pydantic models
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")

    # Common non-serializable types
    if isinstance(value, datetime | date | time):
        return value.isoformat()

    if isinstance(value, timedelta):
        return value.total_seconds()

    if isinstance(value, Decimal):
        return float(value)

    if isinstance(value, dict):
        return {k: to_json_serializable(v) for k, v in value.items()}

    if isinstance(value, list | tuple | set | frozenset):
        return [to_json_serializable(v) for v in value]

    # We have not implemented other cases (e.g. Bytes, Enum, etc.)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON: {value!r}")


@overload
def aggregate_metadata(
    metadata_dict: dict[str, list[Any]], prefix: str = "", return_json_serializable: Literal[True] = True
) -> object: ...


@overload
def aggregate_metadata(
    metadata_dict: dict[str, list[Any]], prefix: str = "", return_json_serializable: Literal[False] = ...
) -> dict: ...


def _collect_all_token_usage(result: dict) -> "TokenUsage":
    """Recursively collect all token_usage from a flattened aggregate_metadata result.

    Args:
        result: The flattened dict from aggregate_metadata (before JSON serialization)

    Returns:
        Combined TokenUsage from all entries (direct and nested sub-agents)
    """
    total = TokenUsage()

    for key, value in result.items():
        if key == "token_usage" and isinstance(value, TokenUsage):
            # Direct token_usage at this level
            total = total + value
        elif isinstance(value, dict):
            # This could be a sub-agent's tool dict - check for nested token_usage
            nested_token_usage = value.get("token_usage")
            if isinstance(nested_token_usage, TokenUsage):
                total = total + nested_token_usage

    return total


def aggregate_metadata(
    metadata_dict: dict[str, list[Any]], prefix: str = "", return_json_serializable: bool = True
) -> dict | object:
    """Aggregate metadata lists and flatten sub-agents into a single-level dict with hierarchical keys.

    For entries with nested run_metadata (e.g., SubAgentMetadata), flattens sub-agents using dot notation.
    Each sub-agent's value is a dict mapping its direct tool names to their aggregated metadata
    (excluding nested sub-agent data, which gets its own top-level key).

    At the root level, token_usage is rolled up to include all sub-agent token usage.

    Args:
        metadata_dict: Dict mapping names (tools or agents) to lists of metadata instances
        prefix: Key prefix for nested calls (used internally for recursion)

    Returns:
        Flat dict with dot-notation keys for sub-agents.
        Example: {
            "token_usage": <combined from all agents>,
            "web_browsing_sub_agent": {"web_search": <aggregated>, "token_usage": <aggregated>},
            "web_browsing_sub_agent.web_fetch_sub_agent": {"fetch_web_page": <aggregated>, "token_usage": <aggregated>}
        }
    """
    result: dict = {}

    # First pass: aggregate all entries in this level (skip internal keys prefixed with _)
    aggregated_level: dict = {}
    for name, metadata_list in metadata_dict.items():
        if name.startswith("_") or not metadata_list:
            continue
        aggregated_level[name] = _aggregate_list(metadata_list)

    # Second pass: separate nested sub-agents from direct tools, and recurse
    direct_tools: dict = {}
    for name, aggregated in aggregated_level.items():
        if hasattr(aggregated, "run_metadata") and isinstance(aggregated.run_metadata, dict):
            # This is a sub-agent - recurse into it
            full_key = f"{prefix}.{name}" if prefix else name
            nested = aggregate_metadata(aggregated.run_metadata, prefix=full_key, return_json_serializable=False)
            result.update(nested)
        else:
            # This is a direct tool/metadata - keep it at this level
            direct_tools[name] = aggregated

    # Store direct tools under the current prefix
    if prefix:
        result[prefix] = direct_tools
    else:
        # At root level, merge direct tools into result
        result.update(direct_tools)

    # At root level, roll up all token_usage from sub-agents
    if not prefix:
        total_token_usage = _collect_all_token_usage(result)
        if total_token_usage.total > 0:
            result["token_usage"] = [total_token_usage]

    if return_json_serializable:
        # Convert all Pydantic models to JSON-serializable dicts
        return to_json_serializable(result)
    return result


# Messages
class TokenUsage(BaseModel):
    """Token counts for LLM usage.

    Token terminology: output = reasoning + answer.
    """

    input: int = 0
    answer: int = 0
    reasoning: int = 0

    @model_validator(mode="before")
    @classmethod
    def _deprecate_output_field(cls, data: object) -> object:
        if isinstance(data, dict) and "output" in data:
            warnings.warn(
                "The 'output' field is deprecated. Use 'answer' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            values = cast("dict[str, Any]", data)
            values.setdefault("answer", values.pop("output"))
        return data

    @property
    def output(self) -> int:
        """Total output tokens (reasoning + answer)."""
        return self.answer + self.reasoning

    @property
    def total(self) -> int:
        """Total token count across input, answer, and reasoning."""
        return self.input + self.answer + self.reasoning

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together, summing each field independently."""
        return TokenUsage(
            input=self.input + other.input,
            answer=self.answer + other.answer,
            reasoning=self.reasoning + other.reasoning,
        )


class ToolUseCountMetadata(BaseModel):
    """Generic metadata tracking tool usage count.

    Implements Addable protocol for aggregation. Use this for tools that only need
    to track how many times they were called.

    Subclasses can override __add__ with their own type thanks to Self typing.
    """

    num_uses: int = 1

    def __add__(self, other: Self) -> Self:
        return self.__class__(num_uses=self.num_uses + other.num_uses)


class ToolResult[M](BaseModel):
    """Result from a tool executor with optional metadata.

    Generic over metadata type M. M should implement Addable protocol for aggregation support,
    but this is not enforced at the class level due to Pydantic schema generation limitations.

    Attributes:
        content: The result content (string, list of content blocks, or images)
        success: Whether the tool call was successful. For finish tools, controls if agent terminates.
        metadata: Optional metadata (e.g., usage stats) that implements Addable for aggregation
    """

    content: Content
    success: bool = True
    metadata: M | None = None


class EmptyParams(BaseModel):
    """Empty parameter model for tools that don't require parameters."""


class Tool[P: BaseModel, M](BaseModel):
    """Tool definition with name, description, parameter schema, and executor function.

    Generic over:
        P: Parameter model type (Pydantic BaseModel subclass, or EmptyParams for parameterless tools)
        M: Metadata type (should implement Addable for aggregation; use None for tools without metadata)

    Tools are simple, stateless callables. For tools requiring lifecycle management
    (setup/teardown, resource pooling), use a ToolProvider instead.

    Example with parameters:
        ```python
        class CalcParams(BaseModel):
            expression: str

        calc_tool = Tool[CalcParams, None](
            name="calc",
            description="Evaluate math",
            parameters=CalcParams,
            executor=lambda p: ToolResult(content=str(eval(p.expression))),
        )
        ```

    Example without parameters (uses EmptyParams by default):
        ```python
        time_tool = Tool[EmptyParams, None](
            name="time",
            description="Get current time",
            executor=lambda _: ToolResult(content=datetime.now().isoformat()),
        )
        ```
    """

    name: str
    description: str
    parameters: type[P] = EmptyParams  # ty: ignore[invalid-assignment]
    executor: Callable[[P], ToolResult[M] | Awaitable[ToolResult[M]]]


class ToolProvider(ABC):
    """Abstract base class for tool providers with lifecycle management.

    ToolProviders manage resources (HTTP clients, sandboxes, server connections)
    and return Tool instances when entering their async context. They implement
    the async context manager protocol.

    Use ToolProvider for:
    - Tools requiring setup/teardown (connections, temp directories)
    - Tools that return multiple Tool instances (e.g., MCP servers)
    - Tools with shared state across calls (e.g., HTTP client pooling)

    Example:
        class MyToolProvider(ToolProvider):
            async def __aenter__(self) -> Tool | list[Tool]:
                # Setup resources and return tool(s)
                return self._create_tool()

            # __aexit__ is optional - default is no-op

    Agent automatically manages ToolProvider lifecycle via its session() context.
    """

    @abstractmethod
    async def __aenter__(self) -> "Tool | list[Tool]":
        """Enter async context: setup resources and return tool(s).

        Returns:
            A single Tool instance, or a list of Tool instances for providers
            that expose multiple tools (e.g., MCP servers).
        """
        ...

    async def __aexit__(  # noqa: B027
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context: cleanup resources. Default: no-op."""


@runtime_checkable
class LLMClient(Protocol):
    """Protocol defining the interface for LLM client implementations.

    Any LLM client must implement this protocol to work with the Agent class.
    Provides text generation with tool support and model capability inspection.
    """

    @abstractmethod
    async def generate(self, messages: list["ChatMessage"], tools: dict[str, Tool]) -> "AssistantMessage": ...

    @property
    def model_slug(self) -> str: ...

    @property
    def max_tokens(self) -> int: ...


class ToolCall(BaseModel, frozen=True):
    """Represents a tool invocation request from the LLM.

    Also a member of the ``AssistantBlock`` union: the ``kind`` discriminator is
    defaulted, so legacy payloads without the key still validate anywhere ToolCall
    is used as a plain input, and new dumps always carry it.

    Attributes:
        name: Name of the tool to invoke
        arguments: JSON string containing tool parameters
        tool_call_id: Unique identifier for tracking this tool call and its result
    """

    kind: Literal["tool_call"] = "tool_call"
    signature: str | None = None
    """Opaque passback state attached to this exact block, e.g. a Google thought signature."""
    name: str
    arguments: str
    tool_call_id: str | None = None


class SystemMessage(BaseModel):
    """System-level instructions and context for the LLM."""

    role: Literal["system"] = "system"
    content: Content


class UserMessage(BaseModel):
    """User input message to the LLM."""

    role: Literal["user"] = "user"
    kind: Literal["user"] = "user"
    content: Content


class SummaryMessage(UserMessage):
    """Summary message bridging summarized-away conversation context.

    Attributes:
        replaced_ids: Ids of the AssistantMessages this summary replaced, so
            consumers of dumped histories can reconstruct lineage offline.
    """

    kind: Literal["summary"] = "summary"
    replaced_ids: list[str] = Field(default_factory=list)


class TurnWarningMessage(UserMessage):
    """Warning message injected when the agent is close to max_turns."""

    kind: Literal["turn_warning"] = "turn_warning"


class Reasoning(BaseModel):
    """Channel-era reasoning shape; survives as the ``AssistantMessage.reasoning`` projection carrier.

    Deprecated as a standalone type: match on the ``AnyReasoningBlock`` kinds
    (ReasoningBlock / SignedReasoningBlock / RedactedReasoningBlock /
    ReasoningRefBlock / EncryptedReasoningBlock) instead.
    """

    signature: str | None = None
    content: str


# Assistant blocks: one assistant turn is an ordered sequence of blocks, in the
# model's actual emission order (thinking → text → thinking → tool call, ...).
class TextBlock(BaseModel, frozen=True):
    """One contiguous run of answer text in an assistant turn.

    ``signature`` carries opaque passback state attached to this exact block,
    e.g. a Google thought signature emitted on a visible text part. It exists
    for external clients only: the built-in clients never set it and drop it
    on replay (they re-emit only ``text``). An external client that sets it is
    responsible for re-emitting it on passback.
    """

    kind: Literal["text"] = "text"
    text: str
    signature: str | None = None


class ReasoningBlock(BaseModel, frozen=True):
    """In-band reasoning text with no passback token.

    E.g. ``reasoning_content`` on Chat Completions-compatible hosts, or
    <think>-tag extraction.
    """

    kind: Literal["reasoning"] = "reasoning"
    content: str


class SignedReasoningBlock(BaseModel, frozen=True):
    """Reasoning bound to an opaque provider signature re-emitted verbatim on passback.

    E.g. Anthropic signed thinking blocks.
    """

    kind: Literal["signed_reasoning"] = "signed_reasoning"
    signature: str
    content: str = ""


class RedactedReasoningBlock(BaseModel, frozen=True):
    """Reasoning the provider withheld, replaced by an opaque payload.

    E.g. Anthropic ``redacted_thinking``: ``data`` must be re-emitted verbatim as a
    ``redacted_thinking`` block on passback. Carries no readable content.
    """

    kind: Literal["redacted_reasoning"] = "redacted_reasoning"
    data: str


class ReasoningRefBlock(BaseModel, frozen=True):
    """Reasoning held provider-side and passed back by reference.

    E.g. OpenAI Responses reasoning items on stored (stateful) conversations:
    the ``id`` is the handle to the server-held item.
    """

    kind: Literal["reasoning_ref"] = "reasoning_ref"
    id: str
    content: str = ""
    """Summary text, when the provider surfaces one."""


class EncryptedReasoningBlock(BaseModel, frozen=True):
    """Reasoning returned as an opaque encrypted payload for stateless passback.

    E.g. OpenAI Responses reasoning items requested with
    ``include: ["reasoning.encrypted_content"]`` (``store=false`` /
    zero-data-retention): the item — id, summary parts, and encrypted payload —
    is re-emitted verbatim in position on passback. The payload is opaque and
    non-inspectable.
    """

    kind: Literal["encrypted_reasoning"] = "encrypted_reasoning"
    id: str
    encrypted_content: str
    summary: tuple[str, ...] = ()
    """Summary parts, when the provider surfaces them; kept split for faithful echo."""

    @property
    def content(self) -> str:
        """Readable summary text for channel projections; parts newline-joined."""
        return _TEXT_BLOCK_SEPARATOR.join(self.summary)


class OpaqueBlock(BaseModel, frozen=True):
    """Provider-native block the framework carries uninterpreted.

    For provider-issued marker/control blocks that must round-trip untouched:
    ``data`` holds the block's raw JSON (self-describing — the provider's own
    ``type`` field travels inside it). The framework preserves it in position
    through history, projections, and serialization so a client that
    understands the payload can re-emit it verbatim on passback; the built-in
    clients do not interpret it and skip it on replay.
    """

    kind: Literal["opaque"] = "opaque"
    data: str


type AnyReasoningBlock = (
    ReasoningBlock | SignedReasoningBlock | RedactedReasoningBlock | ReasoningRefBlock | EncryptedReasoningBlock
)
"""The reasoning family: one kind per passback mechanism."""

REASONING_BLOCK_TYPES = (
    ReasoningBlock,
    SignedReasoningBlock,
    RedactedReasoningBlock,
    ReasoningRefBlock,
    EncryptedReasoningBlock,
)
"""Runtime mirror of ``AnyReasoningBlock`` for isinstance checks — keep in lockstep."""

# No current provider signs *and* references the same reasoning payload; if one
# appears, it becomes a new kind rather than fields grafted onto an existing one.
type AssistantBlock = Annotated[
    TextBlock
    | ReasoningBlock
    | SignedReasoningBlock
    | RedactedReasoningBlock
    | ReasoningRefBlock
    | EncryptedReasoningBlock
    | OpaqueBlock
    | ToolCall
    | ImageContentBlock
    | VideoContentBlock
    | AudioContentBlock,
    Field(discriminator="kind"),
]
"""One block of an assistant turn, discriminated on ``kind``."""

_TEXT_BLOCK_SEPARATOR = "\n"
"""Separator when joining distinct text blocks for the channel projection.

Reasoning content deliberately concatenates *without* a separator — byte-compatible
with what signed passback re-emits."""

type _MediaBlock = ImageContentBlock | VideoContentBlock | AudioContentBlock


def joined_text(blocks: Sequence[AssistantBlock]) -> str | None:
    """All answer text across text blocks, newline-joined; None when there are no text blocks."""
    texts = [block.text for block in blocks if isinstance(block, TextBlock)]
    if not texts:
        return None
    return _TEXT_BLOCK_SEPARATOR.join(texts)


def final_text(blocks: Sequence[AssistantBlock]) -> str | None:
    """Text of the last text block — the "answer" in thinking→text→thinking→text turns."""
    for block in reversed(blocks):
        if isinstance(block, TextBlock):
            return block.text
    return None


def tool_call_blocks(blocks: Sequence[AssistantBlock]) -> list[ToolCall]:
    """Tool calls in emission order."""
    return [block for block in blocks if isinstance(block, ToolCall)]


def reasoning_blocks(blocks: Sequence[AssistantBlock]) -> list[AnyReasoningBlock]:
    """Reasoning blocks (any kind) in emission order."""
    return [block for block in blocks if isinstance(block, REASONING_BLOCK_TYPES)]


def _reasoning_to_block(reasoning: object) -> object:
    """Upgrade a flat channel-era ``Reasoning`` value to its block equivalent.

    A signature means signed passback; bare content is in-band reasoning.
    """
    match reasoning:
        case Reasoning(signature=signature, content=content):
            pass
        case {**mapping}:
            signature = mapping.get("signature")
            content = mapping.get("content")
        case _:
            return reasoning  # let block validation fail loudly
    if signature is not None:
        if not isinstance(signature, str):
            return reasoning
        return SignedReasoningBlock(signature=signature, content=content if isinstance(content, str) else "")
    return ReasoningBlock(content=content if isinstance(content, str) else "")


_CHANNEL_ASSIGNMENT_ERROR = (
    "AssistantMessage.{channel} is a read-only projection of blocks; "
    "use with_text()/with_blocks() or construct a new message. "
    "See the v0.2 migration guide (CHANGELOG.md, BREAKING)."
)
_CHANNEL_PROJECTION_DEPRECATION = (
    "AssistantMessage.{channel} is deprecated; {replacement}. "
    "The compatibility projection will be removed in a future release."
)


def _warn_channel_projection(channel: str, replacement: str) -> None:
    warnings.warn(
        _CHANNEL_PROJECTION_DEPRECATION.format(channel=channel, replacement=replacement),
        DeprecationWarning,
        stacklevel=3,
    )


class AssistantMessage(BaseModel):
    """LLM response message: an ordered sequence of assistant blocks.

    ``blocks`` is the only stored content; the channel-era ``content`` /
    ``reasoning`` / ``tool_calls`` attributes are read-only projections of it.
    Serialized v0.1 payloads upgrade to blocks during validation. Channel-shaped
    construction is not part of the v0.2 API; new code constructs blocks directly.
    Mixing ``blocks`` with non-empty legacy channel keys raises. Use
    ``with_text`` / ``with_blocks`` instead of channel assignment.
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    provider_response_id: str | None = None
    """Provider-attached continuation state, e.g. an OpenAI Responses ``resp_...`` id.

    This is turn metadata rather than emitted assistant content, so it lives beside
    ``blocks`` instead of inside their emission order. It is distinct from ``id``
    (Stirrup's message identity) and ``ReasoningRefBlock.id`` (an emitted reasoning
    item handle).
    """
    role: Literal["assistant"] = "assistant"
    blocks: list[AssistantBlock] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    metadata: dict[str, Any] = Field(default_factory=dict)
    request_start_time: float | None = None
    request_end_time: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _upgrade_channel_fields(cls, data: object) -> object:
        """Convert serialized v0.1 channel payloads into blocks.

        Channel-shaped payloads carry no ordering, so blocks are synthesized in
        the order flat messages have always been replayed: reasoning → text →
        tool calls.
        """
        if not isinstance(data, dict):
            return data
        upgraded = dict(data)
        content = upgraded.pop("content", None)
        reasoning = upgraded.pop("reasoning", None)
        tool_calls = upgraded.pop("tool_calls", None)

        if "blocks" in upgraded:
            # Truthiness on purpose: empty values ("", [], {}) carry no data, so they
            # don't conflict with blocks and are dropped without loss.
            if content or reasoning or tool_calls:
                raise ValueError(
                    "AssistantMessage cannot mix 'blocks' with channel fields "
                    "(content/reasoning/tool_calls); channel data would be silently dropped. "
                    "Pass one representation. See the v0.2 migration guide (CHANGELOG.md, BREAKING)."
                )
            return upgraded

        blocks: list[object] = []
        if reasoning:
            blocks.append(_reasoning_to_block(reasoning))
        if isinstance(content, str):
            if content:
                blocks.append(TextBlock(text=content))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    if item:
                        blocks.append(TextBlock(text=item))
                else:
                    blocks.append(item)  # media blocks already carry their kind discriminator
        elif content is not None:
            raise ValueError(f"AssistantMessage 'content' must be a string or list, got {type(content).__name__}")
        blocks.extend({**tc, "kind": "tool_call"} if isinstance(tc, dict) else tc for tc in tool_calls or [])
        upgraded["blocks"] = blocks
        return upgraded

    # Channel-era projections — read-only, derived from blocks.
    @property
    def content(self) -> Content:
        """Joined text of all text blocks ("" when none); mixed list when media blocks are present."""
        _warn_channel_projection("content", "inspect blocks or use joined_text(message.blocks)")
        if any(isinstance(block, ImageContentBlock | VideoContentBlock | AudioContentBlock) for block in self.blocks):
            return [
                block.text if isinstance(block, TextBlock) else block
                for block in self.blocks
                if isinstance(block, TextBlock | ImageContentBlock | VideoContentBlock | AudioContentBlock)
            ]
        return joined_text(self.blocks) or ""

    @property
    def reasoning(self) -> Reasoning | None:
        """Merged channel view: concatenated reasoning content plus the first signature.

        Redacted blocks carry no readable content and no channel-era equivalent;
        they contribute nothing here.
        """
        _warn_channel_projection("reasoning", "use reasoning_blocks(message.blocks)")
        rblocks = reasoning_blocks(self.blocks)
        if not rblocks:
            return None
        signature = next(
            (block.signature for block in rblocks if isinstance(block, SignedReasoningBlock)),
            None,
        )
        content = "".join(block.content for block in rblocks if not isinstance(block, RedactedReasoningBlock))
        return Reasoning(signature=signature, content=content)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool calls in emission order."""
        _warn_channel_projection("tool_calls", "use tool_call_blocks(message.blocks)")
        return tool_call_blocks(self.blocks)

    @content.setter
    def content(self, _value: object) -> None:
        raise AttributeError(_CHANNEL_ASSIGNMENT_ERROR.format(channel="content"))

    @reasoning.setter
    def reasoning(self, _value: object) -> None:
        raise AttributeError(_CHANNEL_ASSIGNMENT_ERROR.format(channel="reasoning"))

    @tool_calls.setter
    def tool_calls(self, _value: object) -> None:
        raise AttributeError(_CHANNEL_ASSIGNMENT_ERROR.format(channel="tool_calls"))

    def with_text(self, text: str) -> "AssistantMessage":
        """Copy with all text blocks replaced by one text block carrying ``text``.

        The replacement sits where channel-era text sat: before the first tool call.
        Signed text cannot be rewritten because its signature is bound to the exact block.
        ``provider_response_id`` is cleared: the handle is bound to the exact emitted
        content, which the copy no longer carries — keeping it would make stateful
        replay silently substitute the provider's stored (unedited) turn.
        """
        if any(isinstance(block, TextBlock) and block.signature is not None for block in self.blocks):
            raise ValueError("Cannot replace signed text blocks; their signatures are bound to the original text")
        replaced: list[AssistantBlock] = [block for block in self.blocks if not isinstance(block, TextBlock)]
        insert_at = next((i for i, block in enumerate(replaced) if isinstance(block, ToolCall)), len(replaced))
        replaced.insert(insert_at, TextBlock(text=text))
        return self.model_copy(update={"blocks": replaced, "provider_response_id": None})

    def with_blocks(self, blocks: Sequence[AssistantBlock]) -> "AssistantMessage":
        """Copy with ``blocks`` as the new block list; clears ``provider_response_id`` (see ``with_text``)."""
        return self.model_copy(update={"blocks": list(blocks), "provider_response_id": None})

    @property
    def e2e_otps(self) -> float | None:
        """End-to-end output tokens per second."""
        if self.request_start_time is None or self.request_end_time is None:
            return None
        duration = self.request_end_time - self.request_start_time
        if duration <= 0 or self.token_usage.output <= 0:
            return None
        return self.token_usage.output / duration


class ToolMessage(BaseModel):
    """Tool execution result returned to the LLM.

    Attributes:
        role: Always "tool"
        content: The tool result content
        tool_call_id: ID linking this result to the corresponding tool call
        name: Name of the tool that was called
        args_was_valid: Whether the tool arguments were valid
        success: Whether the tool executed successfully (used by finish tool to control termination)
    """

    role: Literal["tool"] = "tool"
    content: Content
    tool_call_id: str | None = None
    name: str | None = None
    args_was_valid: bool = True
    success: bool = False
    tool_start_time: float | None = None
    tool_end_time: float | None = None

    @property
    def tool_duration(self) -> float | None:
        """Tool execution duration in seconds."""
        if self.tool_start_time is None or self.tool_end_time is None:
            return None
        duration = self.tool_end_time - self.tool_start_time
        if duration < 0:
            return None
        return duration


type UserRoleMessage = Annotated[
    UserMessage | SummaryMessage | TurnWarningMessage,
    Field(discriminator="kind"),
]
"""User-role messages, discriminated on ``kind`` — the agent-injected ``UserMessage``
subclasses share ``role="user"``, so dumped histories need the nested discriminator
to rehydrate them as their own types (e.g. ``SummaryMessage.replaced_ids``)."""

type ChatMessage = Annotated[
    SystemMessage | UserRoleMessage | AssistantMessage | ToolMessage, Field(discriminator="role")
]
"""Discriminated union of all message types, automatically parsed based on role field."""


class SubAgentMetadata(BaseModel):
    """Metadata from sub-agent execution including token usage, message history, and child run metadata.

    Implements Addable protocol to support aggregation across multiple subagent calls.
    """

    message_history: list[list[ChatMessage]]
    run_metadata: Annotated[dict[str, Any], Field(default_factory=dict)]

    def __add__(self, other: "SubAgentMetadata") -> "SubAgentMetadata":
        """Combine metadata from multiple subagent calls."""
        # Concatenate message histories
        combined_history = self.message_history + other.message_history
        # Merge run metadata (concatenate lists per key, keep last for non-list internal keys)
        combined_meta: dict[str, Any] = dict(self.run_metadata)
        for key, value in other.run_metadata.items():
            if key in combined_meta and isinstance(combined_meta[key], list) and isinstance(value, list):
                combined_meta[key] = combined_meta[key] + value
            else:
                combined_meta[key] = value
        return SubAgentMetadata(
            message_history=combined_history,
            run_metadata=combined_meta,
        )
