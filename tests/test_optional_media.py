"""Check screenshot-only installs and optional audio/video support."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_images_work_without_moviepy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import io
import sys

class NoMoviePy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "moviepy" or fullname.startswith("moviepy."):
            raise ModuleNotFoundError("MoviePy is absent", name="moviepy")

sys.meta_path.insert(0, NoMoviePy())
from PIL import Image
from stirrup import ImageContentBlock, AudioContentBlock, VideoContentBlock

buffer = io.BytesIO()
Image.new("RGB", (2, 2)).save(buffer, format="PNG")
assert ImageContentBlock(data=buffer.getvalue()).to_base64_url().startswith("data:image/png;base64,")
assert "moviepy" not in sys.modules

for cls in (AudioContentBlock, VideoContentBlock):
    block = cls.model_construct(data=b"")
    for operation in (block._probe, block.to_base64_url):
        try:
            operation()
        except ImportError as exc:
            assert "stirrup[media]" in str(exc)
        else:
            raise AssertionError("Expected installation guidance")
""",
        ],
        check=True,
    )


def test_media_extra_transcodes_audio_and_video(tmp_path: Path) -> None:
    pytest.importorskip("moviepy")
    import wave
    from io import BytesIO

    from moviepy import ColorClip

    from stirrup import AudioContentBlock, VideoContentBlock

    audio = BytesIO()
    with wave.open(audio, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(8000)
        stream.writeframes(b"\0\0" * 4000)
    assert AudioContentBlock(data=audio.getvalue()).to_base64_url().startswith("data:audio/mpeg;base64,")

    video = tmp_path / "sample.mp4"
    with ColorClip(size=(16, 16), color=(0, 0, 0), duration=0.5) as clip:
        clip.write_videofile(str(video), fps=10, codec="libx264", logger=None)
    assert VideoContentBlock(data=video.read_bytes()).to_base64_url(fps=10).startswith("data:video/mp4;base64,")
