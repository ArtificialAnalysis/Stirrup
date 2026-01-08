import base64
import mimetypes
import warnings
from abc import ABC, abstractmethod
from base64 import b64encode
from collections.abc import Awaitable, Callable
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import BytesIO
from math import isinf, isnan, sqrt
from tempfile import NamedTemporaryFile
from types import TracebackType
from typing import Annotated, Any, ClassVar, Literal, Protocol, Self, overload, runtime_checkable

import filetype
from moviepy import AudioFileClip, VideoFileClip
from moviepy.video.fx import Resize
from PIL import Image
from pydantic import BaseModel, Field, PlainSerializer, PlainValidator, model_validator

from stirrup.constants import RESOLUTION_1MP, RESOLUTION_480P

__all__ = [
    "Addable",
    "AssistantMessage",
    "AudioContentBlock",
    "BinaryContentBlock",
    "ChatMessage",
    "Content",
    "ContentBlock",
    "EmptyParams",
    "ImageContentBlock",
    "LLMClient",
    "SubAgentMetadata",
    "SystemMessage",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolMessage",
    "ToolProvider",
    "ToolResult",
    "ToolUseCountMetadata",
    "UserMessage",
    "VideoContentBlock",
    "aggregate_metadata",
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
class BinaryContentBlock(BaseModel, ABC):
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
            raise ValueError("Unsupported mime_type {self.mime_type!r}; allowed: {allowed}")
        self._probe()  # light corruption check; no heavy work
        return self

    @abstractmethod
    def _probe(self) -> None:
        """Verify content can be opened and read; subclasses implement format-specific checks."""


class ImageContentBlock(BinaryContentBlock):
    """Image content supporting PNG, JPEG, WebP, PSD formats with automatic downscaling."""

    kind: Literal["image_content_block"] = "image_content_block"
    allowed_mime_types: ClassVar[set[str]] = {
        "image/jpeg",  # JPEG
        "image/png",  # PNG
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


class VideoContentBlock(BinaryContentBlock):
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


class AudioContentBlock(BinaryContentBlock):
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


def _aggregate_list[T: Addable](metadata_list: list[T]) -> T | None:
    """Aggregate a list of metadata using __add__."""
    if not metadata_list:
        return None
    aggregated = metadata_list[0]
    for m in metadata_list[1:]:
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

    # First pass: aggregate all entries in this level
    aggregated_level: dict = {}
    for name, metadata_list in metadata_dict.items():
        if not metadata_list:
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
    """Token counts for LLM usage (input, output, reasoning tokens)."""

    input: int = 0
    output: int = 0
    reasoning: int = 0

    @property
    def total(self) -> int:
        """Total token count across input, output, and reasoning."""
        return self.input + self.output + self.reasoning

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two TokenUsage objects together, summing each field independently."""
        return TokenUsage(
            input=self.input + other.input,
            output=self.output + other.output,
            reasoning=self.reasoning + other.reasoning,
        )


class ToolUseCountMetadata(BaseModel):
    """Generic metadata tracking tool usage count.

    Implements Addable protocol for aggregation. Use this for tools that only need
    to track how many times they were called.
    """

    num_uses: int = 1

    def __add__(self, other: "ToolUseCountMetadata") -> "ToolUseCountMetadata":
        return ToolUseCountMetadata(num_uses=self.num_uses + other.num_uses)


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
    parameters: type[P] = EmptyParams  # type: ignore[assignment]
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


class ToolCall(BaseModel):
    """Represents a tool invocation request from the LLM.

    Attributes:
        name: Name of the tool to invoke
        arguments: JSON string containing tool parameters
        tool_call_id: Unique identifier for tracking this tool call and its result
    """

    signature: str | None = None
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
    content: Content


class Reasoning(BaseModel):
    """Extended thinking/reasoning content from models that support chain-of-thought reasoning."""

    signature: str | None = None
    content: str


class AssistantMessage(BaseModel):
    """LLM response message with optional tool calls and token usage tracking."""

    role: Literal["assistant"] = "assistant"
    reasoning: Reasoning | None = None
    content: Content
    tool_calls: Annotated[list[ToolCall], Field(default_factory=list)]
    token_usage: Annotated[TokenUsage, Field(default_factory=TokenUsage)]


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


type ChatMessage = Annotated[SystemMessage | UserMessage | AssistantMessage | ToolMessage, Field(discriminator="role")]
"""Discriminated union of all message types, automatically parsed based on role field."""


class SubAgentMetadata(BaseModel):
    """Metadata from sub-agent execution including token usage, message history, and child run metadata.

    Implements Addable protocol to support aggregation across multiple subagent calls.
    """

    message_history: list[list[ChatMessage]]
    run_metadata: Annotated[dict[str, list[Any]], Field(default_factory=dict)]

    def __add__(self, other: "SubAgentMetadata") -> "SubAgentMetadata":
        """Combine metadata from multiple subagent calls."""
        # Concatenate message histories
        combined_history = self.message_history + other.message_history
        # Merge run metadata (concatenate lists per key)
        combined_meta: dict[str, list[Any]] = dict(self.run_metadata)
        for key, metadata_list in other.run_metadata.items():
            if key in combined_meta:
                combined_meta[key] = combined_meta[key] + metadata_list
            else:
                combined_meta[key] = list(metadata_list)
        return SubAgentMetadata(
            message_history=combined_history,
            run_metadata=combined_meta,
        )
