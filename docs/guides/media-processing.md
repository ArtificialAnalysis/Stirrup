# Media Processing

Stirrup keeps media content, processing policy, and provider serialization separate:

```text
ContentBlock → MediaProcessor → provider serializer
```

Clients use `DefaultMediaProcessor` unless another processor is supplied. Its defaults
preserve Stirrup v0.2 behavior:

- Images are limited to 1,000,000 pixels and encoded as RGB PNG.
- Videos are limited to 640 × 480's pixel area and encoded as H.264/AAC MP4.
- Audio is encoded as 192 kbit/s MP3.

Video and audio support is optional:

```bash
pip install "stirrup[media]"
```

The extra provides an FFmpeg binary through `imageio-ffmpeg`. Stirrup loads it only
when video or audio is constructed or processed, so image-only installations do not
download or import the media backend. MoviePy and its larger Python dependency chain
are not required.

Configure those limits when constructing a client:

```python
from stirrup import DefaultMediaProcessor
from stirrup.clients import ChatCompletionsClient

media = DefaultMediaProcessor(
    image_max_pixels=2_000_000,
    video_max_pixels=1280 * 720,
    video_fps=24,
)

client = ChatCompletionsClient(
    model="gpt-4o",
    context_window_tokens=128_000,
    media_processor=media,
)
```

Set an image or video pixel limit to `None` to disable pixel-limit downscaling.
Format normalization and Base64 conversion still occur, and video dimensions are
still made even for codec compatibility.

Implement the `MediaProcessor` protocol to supply a different strategy. A processor
receives an immutable `BinaryContentBlock` and returns `ProcessedMedia`; provider
serializers remain responsible only for their wire format.

## Caching repeated conversions

Wrap a deterministic processor when the same media may be serialized across multiple
model turns:

```python
from stirrup import CachingMediaProcessor, DefaultMediaProcessor
from stirrup.clients import ChatCompletionsClient

with CachingMediaProcessor(
    DefaultMediaProcessor(),
    max_bytes=256 * 1024 * 1024,
) as media:
    client = ChatCompletionsClient(
        model="gpt-4o",
        context_window_tokens=128_000,
        media_processor=media,
    )
    # Use the client while the cache is open.
```

The cache:

- uses the media type, byte length, and SHA-256 digest as its key;
- stores processed output in a private temporary directory;
- evicts least-recently-used files to stay within `max_bytes`;
- ensures concurrent requests process a cache miss only once;
- never stores failed conversions.

Leaving the context or calling `close()` waits for active conversions and removes the
entire directory. `TemporaryDirectory` also provides process-exit cleanup as a fallback.
The code that creates a caching processor owns its lifecycle.

The existing `ImageContentBlock.to_base64_url()`,
`VideoContentBlock.to_base64_url()`, and `AudioContentBlock.to_base64_url()` methods
remain available and delegate to a v0.2-compatible default processor.
