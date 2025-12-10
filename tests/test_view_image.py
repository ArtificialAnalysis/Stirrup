"""Tests for view_image functionality in execution environments."""

import inspect
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from stirrup.core.models import ImageContentBlock, ToolUseCountMetadata
from stirrup.tools.code_backends.base import ViewImageParams
from stirrup.tools.code_backends.local import LocalCodeExecToolProvider


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Create valid PNG image bytes using PIL (1x1 red pixel)."""
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """Create valid JPEG image bytes using PIL (1x1 blue pixel)."""
    img = Image.new("RGB", (1, 1), color=(0, 0, 255))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def sample_image_file(tmp_path: Path, sample_png_bytes: bytes) -> Path:
    """Create a sample PNG image file for tests."""
    file_path = tmp_path / "test_image.png"
    file_path.write_bytes(sample_png_bytes)
    return file_path


@pytest.fixture
def sample_jpeg_file(tmp_path: Path, sample_jpeg_bytes: bytes) -> Path:
    """Create a sample JPEG image file for tests."""
    file_path = tmp_path / "test_image.jpg"
    file_path.write_bytes(sample_jpeg_bytes)
    return file_path


class TestViewImageParams:
    """Tests for ViewImageParams model."""

    def test_valid_path(self) -> None:
        """Test that valid paths are accepted."""
        params = ViewImageParams(path="images/photo.png")
        assert params.path == "images/photo.png"

    def test_absolute_path(self) -> None:
        """Test that absolute paths are accepted."""
        params = ViewImageParams(path="/tmp/image.jpg")
        assert params.path == "/tmp/image.jpg"


class TestImageContentBlock:
    """Tests for ImageContentBlock model."""

    def test_valid_png(self, sample_png_bytes: bytes) -> None:
        """Test that valid PNG data creates ImageContentBlock."""
        block = ImageContentBlock(data=sample_png_bytes)
        assert block.mime_type == "image/png"
        assert block.extension == "png"

    def test_valid_jpeg(self, sample_jpeg_bytes: bytes) -> None:
        """Test that valid JPEG data creates ImageContentBlock."""
        block = ImageContentBlock(data=sample_jpeg_bytes)
        assert block.mime_type == "image/jpeg"
        assert block.extension in ("jpg", "jpeg")

    def test_invalid_data_raises_error(self) -> None:
        """Test that invalid/corrupt data raises ValidationError."""
        with pytest.raises(ValidationError):
            ImageContentBlock(data=b"not an image")

    def test_to_base64_url(self, sample_png_bytes: bytes) -> None:
        """Test base64 URL conversion."""
        block = ImageContentBlock(data=sample_png_bytes)
        url = block.to_base64_url()
        assert url.startswith("data:image/png;base64,")


class TestLocalCodeExecViewImage:
    """Tests for view_image functionality in LocalCodeExecToolProvider."""

    async def test_view_image_relative_path(self, sample_image_file: Path) -> None:
        """Test viewing an image with a relative path."""
        provider = LocalCodeExecToolProvider()

        async with provider:
            # Copy image to temp directory
            assert provider._temp_dir is not None  # noqa: SLF001
            dest_path = provider._temp_dir / "test_image.png"  # noqa: SLF001
            dest_path.write_bytes(sample_image_file.read_bytes())

            # View image using relative path
            result = await provider.view_image("test_image.png")
            assert isinstance(result, ImageContentBlock)
            assert result.mime_type == "image/png"

    async def test_view_image_not_found(self) -> None:
        """Test that viewing non-existent image raises FileNotFoundError."""
        provider = LocalCodeExecToolProvider()

        async with provider:
            with pytest.raises(FileNotFoundError, match="File not found"):
                await provider.view_image("nonexistent.png")

    async def test_view_image_jpeg(self, sample_jpeg_file: Path) -> None:
        """Test viewing a JPEG image."""
        provider = LocalCodeExecToolProvider()

        async with provider:
            assert provider._temp_dir is not None  # noqa: SLF001
            dest_path = provider._temp_dir / "test.jpg"  # noqa: SLF001
            dest_path.write_bytes(sample_jpeg_file.read_bytes())

            result = await provider.view_image("test.jpg")
            assert isinstance(result, ImageContentBlock)
            assert result.mime_type == "image/jpeg"


class TestViewImageTool:
    """Tests for the view_image tool wrapper."""

    async def test_get_view_image_tool_success(self, sample_image_file: Path) -> None:
        """Test successful image viewing through the tool."""
        provider = LocalCodeExecToolProvider()

        async with provider:
            tool = provider.get_view_image_tool()

            # Copy image to temp directory
            assert provider._temp_dir is not None  # noqa: SLF001
            dest_path = provider._temp_dir / "test_image.png"  # noqa: SLF001
            dest_path.write_bytes(sample_image_file.read_bytes())

            # Execute tool
            params = ViewImageParams(path="test_image.png")
            executor_result = tool.executor(params)
            result = await executor_result if inspect.isawaitable(executor_result) else executor_result

            assert isinstance(result.content, list)
            assert len(result.content) == 2
            assert "Viewing image at path" in result.content[0]
            assert isinstance(result.content[1], ImageContentBlock)
            assert isinstance(result.metadata, ToolUseCountMetadata)
            assert result.metadata.num_uses == 1

    async def test_get_view_image_tool_not_found(self) -> None:
        """Test tool handles file not found gracefully."""
        provider = LocalCodeExecToolProvider()

        async with provider:
            tool = provider.get_view_image_tool()

            params = ViewImageParams(path="nonexistent.png")
            executor_result = tool.executor(params)
            result = await executor_result if inspect.isawaitable(executor_result) else executor_result

            assert isinstance(result.content, str)
            assert "not found" in result.content.lower()
            assert isinstance(result.metadata, ToolUseCountMetadata)

    async def test_view_image_tool_metadata_aggregation(self, sample_image_file: Path) -> None:
        """Test that metadata can be aggregated across multiple calls."""
        provider = LocalCodeExecToolProvider()

        async with provider:
            tool = provider.get_view_image_tool()

            # Copy image to temp directory
            assert provider._temp_dir is not None  # noqa: SLF001
            dest_path = provider._temp_dir / "test_image.png"  # noqa: SLF001
            dest_path.write_bytes(sample_image_file.read_bytes())

            # Execute tool twice
            params = ViewImageParams(path="test_image.png")
            executor_result1 = tool.executor(params)  # ty: ignore[too-many-positional-arguments]
            result1 = await executor_result1 if inspect.isawaitable(executor_result1) else executor_result1
            executor_result2 = tool.executor(params)  # ty: ignore[too-many-positional-arguments]
            result2 = await executor_result2 if inspect.isawaitable(executor_result2) else executor_result2

            # Aggregate metadata
            assert result1.metadata is not None
            assert result2.metadata is not None
            combined = result1.metadata + result2.metadata
            assert combined.num_uses == 2


class TestViewImageToolProvider:
    """Tests for ViewImageToolProvider."""

    async def test_auto_detect_exec_env(self) -> None:
        """Test ViewImageToolProvider auto-detects exec_env from session state."""
        from stirrup import Agent
        from stirrup.tools import LocalCodeExecToolProvider, ViewImageToolProvider
        from tests.test_agent import MockLLMClient

        client = MockLLMClient([])
        agent = Agent(
            client=client,
            name="test_agent",
            tools=[
                ViewImageToolProvider(),  # Listed first - tests order independence
                LocalCodeExecToolProvider(),
            ],
        )

        async with agent.session() as session:
            # Verify both tools are available
            assert "code_exec" in session._active_tools  # noqa: SLF001
            assert "view_image" in session._active_tools  # noqa: SLF001

    async def test_explicit_exec_env(self) -> None:
        """Test ViewImageToolProvider with explicit exec_env."""
        from stirrup import Agent
        from stirrup.tools import LocalCodeExecToolProvider, ViewImageToolProvider
        from tests.test_agent import MockLLMClient

        exec_env = LocalCodeExecToolProvider()
        client = MockLLMClient([])
        agent = Agent(
            client=client,
            name="test_agent",
            tools=[
                exec_env,
                ViewImageToolProvider(exec_env),
            ],
        )

        async with agent.session() as session:
            assert "code_exec" in session._active_tools  # noqa: SLF001
            assert "view_image" in session._active_tools  # noqa: SLF001

    async def test_mismatched_exec_env_raises_error(self) -> None:
        """Test that mismatched exec_env raises ValueError."""
        from stirrup import Agent
        from stirrup.tools import LocalCodeExecToolProvider, ViewImageToolProvider
        from tests.test_agent import MockLLMClient

        exec_env1 = LocalCodeExecToolProvider()
        exec_env2 = LocalCodeExecToolProvider()  # Different instance

        client = MockLLMClient([])
        agent = Agent(
            client=client,
            name="test_agent",
            tools=[
                exec_env1,
                ViewImageToolProvider(exec_env2),  # Different exec_env
            ],
        )

        with pytest.raises(ValueError, match="does not match Agent's exec_env"):
            async with agent.session():
                pass

    async def test_no_exec_env_raises_error(self) -> None:
        """Test that missing exec_env raises RuntimeError."""
        from stirrup import Agent
        from stirrup.tools import ViewImageToolProvider, WebToolProvider
        from tests.test_agent import MockLLMClient

        client = MockLLMClient([])
        agent = Agent(
            client=client,
            name="test_agent",
            tools=[
                ViewImageToolProvider(),  # No CodeExecToolProvider
                WebToolProvider(),
            ],
        )

        with pytest.raises(RuntimeError, match="requires a CodeExecToolProvider"):
            async with agent.session():
                pass

    async def test_multiple_exec_env_raises_error(self) -> None:
        """Test that multiple CodeExecToolProviders raise ValueError."""
        from stirrup import Agent
        from stirrup.tools import LocalCodeExecToolProvider
        from tests.test_agent import MockLLMClient

        client = MockLLMClient([])
        agent = Agent(
            client=client,
            name="test_agent",
            tools=[
                LocalCodeExecToolProvider(),
                LocalCodeExecToolProvider(),  # Second exec_env
            ],
        )

        with pytest.raises(ValueError, match="can only have one CodeExecToolProvider"):
            async with agent.session():
                pass
