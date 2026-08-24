"""Configurable media processing, separate from message storage and provider transport."""

import os
from base64 import b64encode
from collections import OrderedDict
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from threading import Lock
from types import TracebackType
from typing import Protocol

from PIL import Image

from stirrup.constants import RESOLUTION_1MP, RESOLUTION_480P
from stirrup.core.models import (
    AudioContentBlock,
    BinaryContentBlock,
    ImageContentBlock,
    VideoContentBlock,
    downscale_image,
)
from stirrup.utils.ffmpeg import transcode_audio, transcode_video, video_dimensions

__all__ = [
    "DEFAULT_MEDIA_PROCESSOR",
    "CachingMediaProcessor",
    "DefaultMediaProcessor",
    "MediaProcessor",
    "ProcessedMedia",
]


@dataclass(frozen=True)
class ProcessedMedia:
    """Encoded media ready for a provider serializer."""

    data: bytes
    mime_type: str
    extension: str

    @property
    def data_url(self) -> str:
        """Return the encoded bytes as a base64 data URL."""
        return f"data:{self.mime_type};base64,{b64encode(self.data).decode()}"


class MediaProcessor(Protocol):
    """Policy boundary between immutable content blocks and provider serializers."""

    def process(self, block: BinaryContentBlock) -> ProcessedMedia:
        """Transform a media block into a provider-ready encoding."""
        ...


@dataclass(frozen=True)
class _CacheEntry:
    path: Path
    size: int
    mime_type: str
    extension: str


class CachingMediaProcessor:
    """Bounded, temporary disk cache around any deterministic media processor."""

    def __init__(self, processor: MediaProcessor, *, max_bytes: int = 256 * 1024 * 1024) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._processor = processor
        self._max_bytes = max_bytes
        self._directory = TemporaryDirectory(prefix="stirrup-media-")
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._inflight: dict[str, Future[ProcessedMedia]] = {}
        self._total_bytes = 0
        self._closed = False
        self._lock = Lock()
        self._close_lock = Lock()

    @property
    def cache_directory(self) -> Path:
        """Private temporary directory removed by ``close()``."""
        return Path(self._directory.name)

    def process(self, block: BinaryContentBlock) -> ProcessedMedia:
        """Return cached output or process one cache miss exactly once."""
        key = self._key(block)
        with self._lock:
            if self._closed:
                raise RuntimeError("CachingMediaProcessor is closed")
            cached = self._read(key)
            if cached is not None:
                return cached
            future = self._inflight.get(key)
            owns_processing = future is None
            if future is None:
                future = self._inflight[key] = Future()

        if not owns_processing:
            return future.result()

        try:
            processed = self._processor.process(block)
            self._store(key, processed)
            future.set_result(processed)
            return processed
        except Exception as error:
            future.set_exception(error)
            raise
        finally:
            if not future.done():
                future.cancel()
            with self._lock:
                self._inflight.pop(key, None)

    def close(self) -> None:
        """Wait for active conversions and remove every cached file."""
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
                self._closed = True
                active = list(self._inflight.values())
            for future in active:
                with suppress(Exception):
                    future.result()
            self._directory.cleanup()

    def __enter__(self) -> "CachingMediaProcessor":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def _key(block: BinaryContentBlock) -> str:
        digest = sha256()
        digest.update(type(block).__qualname__.encode())
        digest.update(len(block.data).to_bytes(8, byteorder="big"))
        digest.update(block.data)
        return digest.hexdigest()

    def _read(self, key: str) -> ProcessedMedia | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        try:
            data = entry.path.read_bytes()
        except FileNotFoundError:
            self._entries.pop(key)
            self._total_bytes -= entry.size
            return None
        self._entries.move_to_end(key)
        return ProcessedMedia(data, mime_type=entry.mime_type, extension=entry.extension)

    def _store(self, key: str, media: ProcessedMedia) -> None:
        size = len(media.data)
        if size > self._max_bytes:
            return

        directory = self.cache_directory
        path = directory / key
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(dir=directory, delete=False) as temporary:
                temporary.write(media.data)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        with self._lock:
            if self._closed:
                path.unlink(missing_ok=True)
                return
            while self._entries and self._total_bytes + size > self._max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._total_bytes -= evicted.size
                evicted.path.unlink(missing_ok=True)
            self._entries[key] = _CacheEntry(
                path=path,
                size=size,
                mime_type=media.mime_type,
                extension=media.extension,
            )
            self._total_bytes += size


@dataclass(frozen=True)
class DefaultMediaProcessor:
    """Stirrup's v0.2 media behavior as an injectable processing policy."""

    image_max_pixels: int | None = RESOLUTION_1MP
    video_max_pixels: int | None = RESOLUTION_480P
    video_fps: int | None = None
    audio_bitrate: str = "192k"

    def process(self, block: BinaryContentBlock) -> ProcessedMedia:
        """Process a supported media block using this policy's limits."""
        if isinstance(block, ImageContentBlock):
            return self._process_image(block)
        if isinstance(block, VideoContentBlock):
            return self._process_video(block)
        if isinstance(block, AudioContentBlock):
            return self._process_audio(block)
        raise TypeError(f"Unsupported media block: {type(block).__name__}")

    def _process_image(self, block: ImageContentBlock) -> ProcessedMedia:
        with Image.open(BytesIO(block.data)) as image:
            if self.image_max_pixels is not None and image.width * image.height > self.image_max_pixels:
                width, height = downscale_image(image.width, image.height, self.image_max_pixels)
                image.thumbnail((width, height), Image.Resampling.LANCZOS)
            if image.mode != "RGB":
                image = image.convert("RGB")
            output = BytesIO()
            image.save(output, format="PNG")
        return ProcessedMedia(output.getvalue(), mime_type="image/png", extension="png")

    def _process_video(self, block: VideoContentBlock) -> ProcessedMedia:
        source_width, source_height = video_dimensions(block.data)
        width, height = downscale_image(source_width, source_height, self.video_max_pixels)
        data = transcode_video(
            block.data,
            width=width,
            height=height,
            fps=self.video_fps,
        )
        return ProcessedMedia(data, mime_type="video/mp4", extension="mp4")

    def _process_audio(self, block: AudioContentBlock) -> ProcessedMedia:
        data = transcode_audio(block.data, bitrate=self.audio_bitrate)
        return ProcessedMedia(data, mime_type="audio/mpeg", extension="mp3")


DEFAULT_MEDIA_PROCESSOR = DefaultMediaProcessor()
"""Default policy preserving Stirrup v0.2's media limits and encodings."""
