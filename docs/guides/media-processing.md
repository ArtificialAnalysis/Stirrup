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

Video and audio operations invoke the FFmpeg binary bundled by `imageio-ffmpeg`
directly; MoviePy and its larger Python dependency chain are not required.

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

The existing `ImageContentBlock.to_base64_url()`,
`VideoContentBlock.to_base64_url()`, and `AudioContentBlock.to_base64_url()` methods
remain available and delegate to a v0.2-compatible default processor.
