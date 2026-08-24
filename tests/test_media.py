"""Tests for the media processing policy boundary."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Event

import pytest
from PIL import Image

from stirrup.clients.open_responses_client import _content_to_open_responses_input
from stirrup.clients.utils import content_to_openai
from stirrup.constants import RESOLUTION_1MP, RESOLUTION_480P
from stirrup.core.media import (
    DEFAULT_MEDIA_PROCESSOR,
    CachingMediaProcessor,
    DefaultMediaProcessor,
    ProcessedMedia,
)
from stirrup.core.models import AudioContentBlock, BinaryContentBlock, ImageContentBlock, VideoContentBlock
from stirrup.utils import ffmpeg as ffmpeg_module
from stirrup.utils.ffmpeg import probe_audio, video_dimensions


def _ffmpeg(*args: str) -> None:
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), *args],
        check=True,
        capture_output=True,
    )


def _image_block(width: int = 1, height: int = 1) -> ImageContentBlock:
    image = Image.new("RGB", (width, height), color=(0, 128, 255))
    output = BytesIO()
    image.save(output, format="PNG")
    return ImageContentBlock(data=output.getvalue())


class RecordingMediaProcessor:
    def __init__(self) -> None:
        self.blocks: list[BinaryContentBlock] = []

    def process(self, block: BinaryContentBlock) -> ProcessedMedia:
        self.blocks.append(block)
        return ProcessedMedia(data=b"custom", mime_type="image/png", extension="png")


class BlockingMediaProcessor(RecordingMediaProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.started = Event()
        self.release = Event()

    def process(self, block: BinaryContentBlock) -> ProcessedMedia:
        self.started.set()
        assert self.release.wait(timeout=5)
        return super().process(block)


class FlakyMediaProcessor(RecordingMediaProcessor):
    def process(self, block: BinaryContentBlock) -> ProcessedMedia:
        if not self.blocks:
            self.blocks.append(block)
            raise ValueError("transcode failed")
        return super().process(block)


def test_missing_media_extra_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(_name: str) -> None:
        raise ModuleNotFoundError

    monkeypatch.setattr(ffmpeg_module, "import_module", missing_import)

    with pytest.raises(ImportError, match=r"stirrup\[media\]"):
        video_dimensions(b"video")


def test_default_policy_preserves_v02_limits() -> None:
    assert DefaultMediaProcessor() == DefaultMediaProcessor(
        image_max_pixels=RESOLUTION_1MP,
        video_max_pixels=RESOLUTION_480P,
        video_fps=None,
        audio_bitrate="192k",
    )


def test_legacy_image_method_delegates_to_default_policy() -> None:
    block = _image_block()

    assert block.to_base64_url() == DEFAULT_MEDIA_PROCESSOR.process(block).data_url


def test_image_limit_is_configurable_without_changing_content_block() -> None:
    block = _image_block(200, 100)

    processed = DefaultMediaProcessor(image_max_pixels=5_000).process(block)
    image = Image.open(BytesIO(processed.data))

    assert image.width * image.height <= 5_000
    assert image.width / image.height == 2
    assert block.mime_type == "image/png"


def test_video_limit_is_configurable_without_changing_content_block(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=32x16:d=0.2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "5",
        "-y",
        str(source),
    )
    block = VideoContentBlock(data=source.read_bytes())

    processed = DefaultMediaProcessor(video_max_pixels=128, video_fps=5).process(block)
    assert video_dimensions(processed.data) == (16, 8)
    assert processed.mime_type == "video/mp4"
    assert block.mime_type == "video/mp4"


def test_audio_bitrate_is_configurable_without_changing_content_block(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    _ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:duration=0.1",
        "-y",
        str(source),
    )
    block = AudioContentBlock(data=source.read_bytes())

    processed = DefaultMediaProcessor(audio_bitrate="64k").process(block)

    probe_audio(processed.data)
    assert processed.mime_type == "audio/mpeg"
    assert processed.extension == "mp3"
    assert block.mime_type == "audio/x-wav"


def test_openai_serializers_use_injected_media_processor() -> None:
    block = _image_block()
    processor = RecordingMediaProcessor()
    expected_url = "data:image/png;base64,Y3VzdG9t"

    chat_content = content_to_openai([block], media_processor=processor)
    responses_content = _content_to_open_responses_input([block], media_processor=processor)

    assert chat_content == [{"type": "image_url", "image_url": {"url": expected_url}}]
    assert responses_content == [{"type": "input_image", "image_url": expected_url}]
    assert processor.blocks == [block, block]


def test_cache_reuses_processed_media_and_cleans_up() -> None:
    block = _image_block()
    processor = RecordingMediaProcessor()
    cache = CachingMediaProcessor(processor)
    directory = cache.cache_directory

    with cache:
        assert cache.process(block) == cache.process(block)
        assert directory.is_dir()
        assert any(directory.iterdir())

    assert processor.blocks == [block]
    assert not directory.exists()
    with pytest.raises(RuntimeError, match="closed"):
        cache.process(block)


def test_cache_evicts_least_recently_used_media_by_total_bytes() -> None:
    first = _image_block()
    second = _image_block(2, 1)
    processor = RecordingMediaProcessor()

    with CachingMediaProcessor(processor, max_bytes=len(b"custom")) as cache:
        cache.process(first)
        cache.process(second)
        cache.process(first)

    assert processor.blocks == [first, second, first]


def test_cache_processes_concurrent_misses_once() -> None:
    block = _image_block()
    processor = BlockingMediaProcessor()

    with CachingMediaProcessor(processor) as cache, ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(cache.process, block)
        assert processor.started.wait(timeout=5)
        second = pool.submit(cache.process, block)
        processor.release.set()

        assert first.result(timeout=5) == second.result(timeout=5)

    assert processor.blocks == [block]


def test_cache_does_not_store_failed_conversions() -> None:
    block = _image_block()
    processor = FlakyMediaProcessor()

    with CachingMediaProcessor(processor) as cache:
        with pytest.raises(ValueError, match="transcode failed"):
            cache.process(block)
        assert cache.process(block).data == b"custom"

    assert processor.blocks == [block, block]
