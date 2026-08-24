"""Configurable media processing, separate from message storage and provider transport."""

from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO
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
