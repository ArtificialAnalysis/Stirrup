"""Small direct-FFmpeg backend for media probing and transcoding."""

import subprocess
from collections.abc import Generator
from importlib import import_module
from tempfile import NamedTemporaryFile
from typing import Any, Protocol, cast


class _ImageIOFFmpeg(Protocol):
    def get_ffmpeg_exe(self) -> str: ...

    def read_frames(self, path: str) -> Generator[dict[str, Any] | bytes, None, None]: ...


def _load_ffmpeg() -> _ImageIOFFmpeg:
    try:
        return cast(_ImageIOFFmpeg, import_module("imageio_ffmpeg"))
    except ModuleNotFoundError as error:
        raise ImportError(
            "Video and audio processing requires the 'media' extra. "
            "Install it with: pip install 'stirrup[media]'"
        ) from error


def _run_ffmpeg(*args: str) -> None:
    ffmpeg = _load_ffmpeg()
    try:
        result = subprocess.run(
            [ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ValueError(f"Could not run FFmpeg: {error}") from error
    if result.returncode != 0:
        message = result.stderr.strip() or "Unknown FFmpeg error"
        raise ValueError(message)


def video_dimensions(data: bytes) -> tuple[int, int]:
    """Read a video's dimensions without decoding it into Python."""
    ffmpeg = _load_ffmpeg()
    with NamedTemporaryFile(suffix=".bin") as source:
        source.write(data)
        source.flush()
        try:
            frames = ffmpeg.read_frames(source.name)
            try:
                metadata = next(frames)
            finally:
                frames.close()
        except (OSError, RuntimeError) as error:
            raise ValueError(f"Could not read video: {error}") from error

    if not isinstance(metadata, dict):
        raise ValueError("FFmpeg returned no video metadata")
    size = metadata.get("size")
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("FFmpeg returned no video dimensions")
    width, height = size
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError("FFmpeg returned invalid video dimensions")
    return width, height


def probe_audio(data: bytes) -> None:
    """Verify that bytes contain a readable audio stream."""
    with NamedTemporaryFile(suffix=".bin") as source:
        source.write(data)
        source.flush()
        _run_ffmpeg(
            "-i",
            source.name,
            "-map",
            "0:a:0",
            "-t",
            "0.01",
            "-f",
            "null",
            "-",
        )


def transcode_video(
    data: bytes,
    *,
    width: int,
    height: int,
    fps: int | None,
) -> bytes:
    """Encode video as H.264/AAC MP4."""
    with NamedTemporaryFile(suffix=".bin") as source, NamedTemporaryFile(suffix=".mp4") as output:
        source.write(data)
        source.flush()
        fps_args = [] if fps is None else ["-r", str(fps)]
        _run_ffmpeg(
            "-i",
            source.name,
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            f"scale={width}:{height}",
            *fps_args,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-y",
            output.name,
        )
        return output.read()


def transcode_audio(data: bytes, *, bitrate: str) -> bytes:
    """Encode audio as MP3."""
    with NamedTemporaryFile(suffix=".bin") as source, NamedTemporaryFile(suffix=".mp3") as output:
        source.write(data)
        source.flush()
        _run_ffmpeg(
            "-i",
            source.name,
            "-map",
            "0:a:0",
            "-vn",
            "-c:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            "-y",
            output.name,
        )
        return output.read()
