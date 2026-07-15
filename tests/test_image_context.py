"""Tests for image context budgeting (resize/compression and bounded history)."""

import inspect
import random
from io import BytesIO

import pytest
from PIL import Image

from stirrup.constants import MAX_IMAGE_BYTES, RESOLUTION_1MP
from stirrup.core.agent import _prune_excess_image_blocks
from stirrup.core.models import ImageContentBlock, ToolMessage, UserMessage, prepare_image_bytes
from stirrup.tools.code_backends.base import ViewImageParams
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider


def _large_image_bytes(*, width: int = 2000, height: int = 2000) -> bytes:
    """Create a large uncompressed BMP image for budget tests."""
    img = Image.new("RGB", (width, height), color=(12, 34, 56))
    buffer = BytesIO()
    img.save(buffer, format="BMP")
    return buffer.getvalue()


def _noisy_image_bytes(*, width: int = 256, height: int = 256) -> bytes:
    """Create a high-entropy image that resists quality-only JPEG compression."""
    img = Image.new("RGB", (width, height))
    rng = random.Random(0)
    for y in range(height):
        for x in range(width):
            img.putpixel((x, y), (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestPrepareImageBytes:
    """Tests for prepare_image_bytes normalization."""

    def test_preserves_small_images(self, sample_png_bytes: bytes) -> None:
        """Small images should pass through unchanged."""
        assert prepare_image_bytes(sample_png_bytes) == sample_png_bytes

    def test_downscales_and_compresses_large_images(self) -> None:
        """Large images should be reduced below configured byte and pixel limits."""
        raw = _large_image_bytes()
        assert len(raw) > MAX_IMAGE_BYTES

        prepared = prepare_image_bytes(raw, max_pixels=RESOLUTION_1MP, max_bytes=MAX_IMAGE_BYTES)
        assert len(prepared) <= MAX_IMAGE_BYTES

        with Image.open(BytesIO(prepared)) as img:
            assert img.width * img.height <= RESOLUTION_1MP

    def test_respects_disabled_limits(self) -> None:
        """Passing None for limits should preserve original bytes."""
        raw = _large_image_bytes(width=10, height=10)
        assert prepare_image_bytes(raw, max_pixels=None, max_bytes=None) == raw

    def test_noisy_image_meets_byte_budget_via_dimension_reduction(self) -> None:
        """Noisy images should still satisfy max_bytes after dimension reduction."""
        raw = _noisy_image_bytes()
        max_bytes = 8_000

        buf = BytesIO()
        with Image.open(BytesIO(raw)) as img:
            rgb = img.convert("RGB")
            rgb.save(buf, format="JPEG", quality=30, optimize=True)
        assert len(buf.getvalue()) > max_bytes

        prepared = prepare_image_bytes(raw, max_pixels=None, max_bytes=max_bytes)
        assert len(prepared) <= max_bytes


class TestViewImageNormalization:
    """Tests for view_image tool ingestion normalization."""

    async def test_view_image_tool_compresses_large_file(self) -> None:
        """view_image should store normalized bytes, not the full source file."""
        provider = LocalCodeExecToolProvider(max_image_bytes=50_000)

        async with provider:
            assert provider._temp_dir is not None  # noqa: SLF001
            image_path = provider._temp_dir / "large.png"  # noqa: SLF001
            image_path.write_bytes(_large_image_bytes())

            tool = provider.get_view_image_tool()
            params = ViewImageParams(path="large.png")
            executor_result = tool.executor(params)
            result = await executor_result if inspect.isawaitable(executor_result) else executor_result

            assert isinstance(result.content, list)
            image_block = result.content[1]
            assert isinstance(image_block, ImageContentBlock)
            assert len(image_block.data) <= 50_000

    async def test_repeated_view_image_calls_stay_bounded(self) -> None:
        """Repeated view_image calls should each return bounded image payloads."""
        provider = LocalCodeExecToolProvider(max_image_bytes=40_000)

        async with provider:
            assert provider._temp_dir is not None  # noqa: SLF001
            image_path = provider._temp_dir / "chart.png"  # noqa: SLF001
            image_path.write_bytes(_large_image_bytes())

            tool = provider.get_view_image_tool()
            params = ViewImageParams(path="chart.png")

            for _ in range(3):
                executor_result = tool.executor(params)
                result = await executor_result if inspect.isawaitable(executor_result) else executor_result
                image_block = result.content[1]
                assert isinstance(image_block, ImageContentBlock)
                assert len(image_block.data) <= 40_000


class TestImageHistoryPruning:
    """Tests for bounded image history before LLM calls."""

    async def test_agent_step_prunes_images_before_generate(self) -> None:
        """Agent.step should not send every historical image block to the model."""
        from stirrup import Agent
        from stirrup.core.models import AssistantMessage, ChatMessage, TokenUsage
        from tests.test_agent import MockLLMClient

        seen_image_blocks = 0

        class CountingClient(MockLLMClient):
            async def generate(
                self,
                messages: list[ChatMessage],
                tools: object,
            ) -> AssistantMessage:
                del tools
                nonlocal seen_image_blocks
                for message in messages:
                    if isinstance(message.content, list):
                        seen_image_blocks += sum(1 for block in message.content if isinstance(block, ImageContentBlock))
                return AssistantMessage(
                    content="ok",
                    tool_calls=[],
                    token_usage=TokenUsage(input=1, answer=1),
                )

        blocks = [ImageContentBlock(data=_large_image_bytes(width=2, height=2)) for _ in range(4)]
        messages = [
            ToolMessage(content=["older", blocks[0]], tool_call_id="call_0", name="view_image"),
            ToolMessage(content=["older", blocks[1]], tool_call_id="call_1", name="view_image"),
            ToolMessage(content=["newer", blocks[2]], tool_call_id="call_2", name="view_image"),
            ToolMessage(content=["newest", blocks[3]], tool_call_id="call_3", name="view_image"),
        ]

        agent = Agent(
            client=CountingClient([]),
            name="test-agent",
            max_turns=1,
            max_images_in_context=2,
            tools=[],
        )

        await agent.step(messages, run_metadata={})
        assert seen_image_blocks == 2

    def test_prune_replaces_oldest_images(self) -> None:
        """Only the newest max_images blocks should remain as images."""
        blocks = [ImageContentBlock(data=_large_image_bytes(width=2, height=2)) for _ in range(4)]
        messages = [
            ToolMessage(
                content=["first", blocks[0]],
                tool_call_id="call_1",
                name="view_image",
            ),
            UserMessage(content=[blocks[1]]),
            ToolMessage(
                content=["second", blocks[2]],
                tool_call_id="call_2",
                name="view_image",
            ),
            UserMessage(content=[blocks[3]]),
        ]

        _prune_excess_image_blocks(messages, max_images=2)

        remaining_images = [
            block
            for message in messages
            if isinstance(message.content, list)
            for block in message.content
            if isinstance(block, ImageContentBlock)
        ]
        assert len(remaining_images) == 2

        first_tool = messages[0]
        assert isinstance(first_tool.content, list)
        assert isinstance(first_tool.content[1], str)
        assert "removed from context" in first_tool.content[1]

    def test_prune_noop_when_disabled(self) -> None:
        """None max_images should leave all image blocks intact."""
        block = ImageContentBlock(data=_large_image_bytes(width=2, height=2))
        messages = [UserMessage(content=[block])]
        _prune_excess_image_blocks(messages, max_images=None)
        content = messages[0].content
        assert isinstance(content, list)
        assert isinstance(content[0], ImageContentBlock)


class TestAgentImageContextConfig:
    """Tests for agent-level image context configuration."""

    def test_rejects_negative_max_images_in_context(self) -> None:
        """max_images_in_context must be non-negative when set."""
        from stirrup import Agent
        from tests.test_agent import MockLLMClient

        with pytest.raises(ValueError, match="max_images_in_context must be non-negative or None"):
            Agent(
                client=MockLLMClient([]),
                name="test-agent",
                max_images_in_context=-1,
            )

    def test_allows_none_max_images_in_context(self) -> None:
        """None should disable image-count pruning."""
        from stirrup import Agent
        from tests.test_agent import MockLLMClient

        agent = Agent(
            client=MockLLMClient([]),
            name="test-agent",
            max_images_in_context=None,
        )
        assert agent._max_images_in_context is None  # noqa: SLF001


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Create valid PNG image bytes using PIL (1x1 red pixel)."""
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()
