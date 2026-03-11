"""Smoke test for MCP image tool results."""

import base64
import inspect
import sys
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from PIL import Image

from stirrup.clients.utils import to_openai_messages
from stirrup.core.models import ImageContentBlock, ToolMessage, ToolResult, ToolUseCountMetadata
from stirrup.tools.mcp import MCPConfig, MCPToolProvider

pytest.importorskip("mcp.server.fastmcp")


def _png_b64() -> str:
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_image_server(script_path: Path) -> None:
    png_b64 = _png_b64()
    script_path.write_text(
        f"""
import base64

from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("image-server")


@mcp.tool()
def read_image() -> Image:
    return Image(data=base64.b64decode("{png_b64}"), format="png")


if __name__ == "__main__":
    mcp.run(transport="stdio")
""".strip()
    )


def _make_provider(script_path: Path) -> MCPToolProvider:
    config = MCPConfig.model_validate(
        {
            "mcpServers": {
                "image_server": {
                    "command": sys.executable,
                    "args": [str(script_path)],
                }
            }
        }
    )
    return MCPToolProvider(config=config)


async def test_mcp_image_result_reaches_openai_message(tmp_path: Path) -> None:
    script_path = tmp_path / "image_server.py"
    _write_image_server(script_path)

    provider = _make_provider(script_path)
    async with provider as tools:
        tool = next(tool for tool in tools if tool.name == "image_server__read_image")
        executor_result = tool.executor(tool.parameters())
        raw_result = await executor_result if inspect.isawaitable(executor_result) else executor_result
        result = cast(ToolResult[ToolUseCountMetadata], raw_result)

    assert isinstance(result.content, list)
    assert len(result.content) == 1
    assert isinstance(result.content[0], ImageContentBlock)

    messages = to_openai_messages(
        [
            ToolMessage(
                content=result.content,
                tool_call_id="call_1",
                name="image_server__read_image",
            )
        ]
    )
    assert messages[0]["content"][0]["type"] == "image_url"
