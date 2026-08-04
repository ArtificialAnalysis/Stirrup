from io import BytesIO
from pathlib import Path
from threading import get_ident

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

    assert isinstance(result.content, str)
    assert len(result.content) == 20
    assert "[truncated]" in result.content


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
    assert isinstance(result.content, list)
    assert isinstance(result.content[0], str)
    assert isinstance(result.content[1], str)
    assert "Spilled 1000 characters to" in result.content[0]
    assert "[truncated]" in result.content[1]
    assert sum(len(block) for block in result.content if isinstance(block, str)) == 500


async def test_truncator_preserves_media_order() -> None:
    image_bytes = BytesIO()
    Image.new("RGB", (2, 2)).save(image_bytes, format="PNG")
    image = ImageContentBlock(data=image_bytes.getvalue())

    async def executor(params: Params) -> ToolResult:
        return ToolResult(content=[params.text, image, params.text])

    tool = Tool(
        name="mixed",
        description="Return image and text",
        parameters=Params,
        executor=executor,
    )
    wrapped = ToolTruncatorMiddleware(max_chars=40)(tool)

    result = await call_executor(wrapped, Params(text="x" * 100))

    assert isinstance(result.content, list)
    assert isinstance(result.content[0], str)
    assert isinstance(result.content[2], str)
    assert result.content[1] is image
    assert "[truncated]" in result.content[0]
    assert sum(len(block) for block in result.content if isinstance(block, str)) == 40


async def test_truncator_rejects_budget_smaller_than_spill_notice(tmp_path: Path) -> None:
    spill = DiskSpillMiddleware(max_chars=20, sink=LocalDirSink(tmp_path))
    truncate = ToolTruncatorMiddleware(max_chars=20)
    tool = truncate(spill(make_tool()))

    with pytest.raises(ValueError, match="too small to preserve the spill notice"):
        await call_executor(tool, Params(text="x" * 100))


async def test_sync_tool_can_run_in_event_loop() -> None:
    thread_ids: list[int] = []

    def executor(params: Params) -> ToolResult:
        thread_ids.append(get_ident())
        return ToolResult(content=params.text)

    tool = Tool(name="sync", description="Sync tool", parameters=Params, executor=executor)
    wrapped = ToolTruncatorMiddleware(max_chars=20, run_sync_in_thread=False)(tool)

    await call_executor(wrapped, Params(text="short"))

    assert thread_ids == [get_ident()]


def test_truncator_requires_room_for_marker() -> None:
    with pytest.raises(ValueError, match="max_chars must be at least"):
        ToolTruncatorMiddleware(max_chars=0)


def test_spill_requires_positive_budget() -> None:
    with pytest.raises(ValueError, match="max_chars must be positive"):
        DiskSpillMiddleware(max_chars=0, sink=LocalDirSink("."))
