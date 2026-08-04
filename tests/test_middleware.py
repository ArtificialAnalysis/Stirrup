from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from pydantic import BaseModel

from stirrup import (
    DiskSpillMiddleware,
    ImageContentBlock,
    LocalDirSink,
    Tool,
    ToolResult,
    ToolTruncatorMiddleware,
)
from stirrup.core.middleware import call_executor


class Params(BaseModel):
    text: str


def make_tool(name: str = "echo") -> Tool:
    return Tool(
        name=name,
        description="Echo text",
        parameters=Params,
        executor=lambda params: ToolResult(content=params.text),
    )


async def test_truncator_wraps_sync_tool() -> None:
    tool = ToolTruncatorMiddleware(max_chars=20)(make_tool())

    result = await call_executor(tool, Params(text="x" * 100))

    assert "This content has been truncated" in result.content


async def test_spill_then_truncate_preserves_full_output(tmp_path: Path) -> None:
    spill = DiskSpillMiddleware(max_chars=500, sink=LocalDirSink(tmp_path))
    truncate = ToolTruncatorMiddleware(max_chars=500)
    tool = truncate(spill(make_tool("../unsafe/tool")))

    result = await call_executor(tool, Params(text="x" * 1_000))
    files = list(tmp_path.iterdir())

    assert len(files) == 1
    assert files[0].parent == tmp_path
    assert ".." not in files[0].name
    assert files[0].read_text() == "x" * 1_000
    assert "has been saved to" in result.content
    assert "This content has been truncated" in result.content


async def test_truncator_preserves_media_order() -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2)).save(image_bytes, format="PNG")
    image = ImageContentBlock(data=image_bytes.getvalue())

    async def executor(params: Params) -> ToolResult:
        return ToolResult(content=[image, params.text])

    tool = Tool(
        name="mixed",
        description="Return image and text",
        parameters=Params,
        executor=executor,
    )
    wrapped = ToolTruncatorMiddleware(max_chars=20)(tool)

    result = await call_executor(wrapped, Params(text="x" * 100))

    assert isinstance(result.content, list)
    assert result.content[0] is image
    assert "This content has been truncated" in result.content[1]


def test_truncator_requires_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_chars must be positive"):
        ToolTruncatorMiddleware(max_chars=0)


def test_spill_requires_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_chars must be positive"):
        DiskSpillMiddleware(max_chars=0, sink=LocalDirSink("."))
