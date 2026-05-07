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


def _image_b64(image_format: str) -> str:
    """Build a tiny real image payload for the temp MCP server."""
    img = Image.new("RGB", (1, 1), color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _write_image_server(script_path: Path, image_format: str, tool_name: str) -> None:
    """Write a one-file stdio MCP server with a single image-returning tool."""
    image_b64 = _image_b64(image_format)
    image_ext = image_format.lower()
    script_path.write_text(
        f"""
import base64

from mcp.server.fastmcp import FastMCP, Image

mcp = FastMCP("image-server")


@mcp.tool()
def {tool_name}() -> Image:
    return Image(data=base64.b64decode("{image_b64}"), format="{image_ext}")


if __name__ == "__main__":
    mcp.run(transport="stdio")
""".strip()
    )


def _make_provider(script_path: Path) -> MCPToolProvider:
    """Create a provider that launches the temp MCP server over stdio."""
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


async def _assert_tool_returns_image(
    tmp_path: Path,
    *,
    image_format: str,
    tool_name: str,
) -> None:
    """Assert the MCP bridge preserves an image through OpenAI-style serialization."""
    script_path = tmp_path / "image_server.py"
    _write_image_server(script_path, image_format=image_format, tool_name=tool_name)

    provider = _make_provider(script_path)
    async with provider as tools:
        tool = next(tool for tool in tools if tool.name == f"image_server__{tool_name}")
        executor_result = tool.executor(tool.parameters())
        raw_result = await executor_result if inspect.isawaitable(executor_result) else executor_result
        result = cast(ToolResult[ToolUseCountMetadata], raw_result)

    # First prove the MCP bridge produced a real Stirrup image block.
    assert isinstance(result.content, list)
    assert len(result.content) == 1
    assert isinstance(result.content[0], ImageContentBlock)
    assert result.content[0].mime_type == f"image/{image_format.lower()}"

    # Then prove the image still survives message serialization for the model layer.
    messages = to_openai_messages(
        [
            ToolMessage(
                content=result.content,
                tool_call_id="call_1",
                name=f"image_server__{tool_name}",
            )
        ]
    )
    assert messages[0]["content"][0]["type"] == "image_url"


async def test_mcp_png_result_reaches_openai_message(tmp_path: Path) -> None:
    await _assert_tool_returns_image(tmp_path, image_format="PNG", tool_name="read_png")


async def test_mcp_webp_result_reaches_openai_message(tmp_path: Path) -> None:
    await _assert_tool_returns_image(tmp_path, image_format="WEBP", tool_name="read_webp")
