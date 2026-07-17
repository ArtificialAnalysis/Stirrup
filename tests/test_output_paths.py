"""Unit tests for output-path resolution and the base output-validation hooks."""

from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from stirrup.core.models import ImageContentBlock
from stirrup.tools.code_backends.base import (
    CodeExecToolProvider,
    CommandResult,
    OutputSourceRoot,
    _safe_output_relative_path,
)

POSIX_ROOT = OutputSourceRoot(path="/workspace")


class TestSafeOutputRelativePath:
    """Source paths are resolved with POSIX semantics (execution envs are Linux)."""

    def test_relative_path_preserves_structure(self) -> None:
        result = _safe_output_relative_path("nested/dir/report.txt", source_roots=(POSIX_ROOT,))
        assert result == PurePosixPath("nested/dir/report.txt")

    def test_absolute_path_inside_root_made_relative(self) -> None:
        result = _safe_output_relative_path("/workspace/out/report.txt", source_roots=(POSIX_ROOT,))
        assert result == PurePosixPath("out/report.txt")

    def test_absolute_path_outside_root_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside known execution roots"):
            _safe_output_relative_path("/etc/passwd", source_roots=(POSIX_ROOT,))

    def test_absolute_path_with_no_roots_rejected(self) -> None:
        with pytest.raises(ValueError, match="outside known execution roots"):
            _safe_output_relative_path("/workspace/report.txt", source_roots=())

    def test_traversal_rejected(self) -> None:
        with pytest.raises(ValueError, match="contains traversal"):
            _safe_output_relative_path("nested/../report.txt", source_roots=(POSIX_ROOT,))

    def test_empty_path_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty or invalid"):
            _safe_output_relative_path("", source_roots=(POSIX_ROOT,))

    def test_nul_byte_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty or invalid"):
            _safe_output_relative_path("report\x00.txt", source_roots=(POSIX_ROOT,))

    def test_dot_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not identify a file"):
            _safe_output_relative_path(".", source_roots=(POSIX_ROOT,))


class _MemoryExecProvider(CodeExecToolProvider):
    """Minimal test backend with an in-memory filesystem."""

    def __init__(self) -> None:
        super().__init__()
        self.files: dict[str, bytes] = {}
        self.written: dict[str, bytes] = {}

    async def __aenter__(self):  # noqa: ANN204
        return self.get_code_exec_tool()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        return None

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        raise NotImplementedError

    async def read_file_bytes(self, path: str) -> bytes:
        return self.files[path]

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        self.written[path] = content

    async def file_exists(self, path: str) -> bool:  # noqa: ARG002
        return False

    async def is_directory(self, path: str) -> bool:  # noqa: ARG002
        return False

    async def list_files(self, path: str) -> list[str]:  # noqa: ARG002
        return []

    async def view_image(self, path: str) -> ImageContentBlock:
        raise NotImplementedError


class TestBaseOutputHookDefaults:
    """The documented extension-point defaults for backends that don't override them."""

    async def test_output_destination_identity_accepts_in_root(self) -> None:
        identity = await _MemoryExecProvider().output_destination_identity("outbox/nested/report.txt", "outbox")
        assert identity == "outbox/nested/report.txt"

    async def test_output_destination_identity_rejects_escape(self) -> None:
        with pytest.raises(ValueError, match="escapes output directory"):
            await _MemoryExecProvider().output_destination_identity("elsewhere/report.txt", "outbox")


class TestCrossEnvironmentTransfer:
    """The cross-env branch of base save_output_files (dest_env set)."""

    async def test_transfer_preserves_nested_paths(self) -> None:
        source = _MemoryExecProvider()
        source.files = {"left/report.txt": b"left", "right/report.txt": b"right"}
        dest = _MemoryExecProvider()

        result = await source.save_output_files(["left/report.txt", "right/report.txt"], "outbox", dest_env=dest)

        assert result.failed == {}
        assert dest.written == {
            "outbox/left/report.txt": b"left",
            "outbox/right/report.txt": b"right",
        }

    async def test_duplicate_destination_rejected_first_claim_holds(self) -> None:
        # "./report.txt" and "report.txt" normalize to one destination; the
        # second spelling must be rejected, not silently overwrite the first.
        source = _MemoryExecProvider()
        source.files = {"report.txt": b"first", "./report.txt": b"second"}
        dest = _MemoryExecProvider()

        result = await source.save_output_files(["report.txt", "./report.txt"], "outbox", dest_env=dest)

        assert [saved.source_path for saved in result.saved] == ["report.txt"]
        assert "collision" in result.failed["./report.txt"]
        assert dest.written == {"outbox/report.txt": b"first"}

    async def test_exact_duplicate_source_is_idempotent(self) -> None:
        source = _MemoryExecProvider()
        source.files = {"report.txt": b"data"}
        dest = _MemoryExecProvider()

        result = await source.save_output_files(["report.txt", "report.txt"], "outbox", dest_env=dest)

        assert result.failed == {}
        assert [saved.source_path for saved in result.saved] == ["report.txt"]
        assert dest.written == {"outbox/report.txt": b"data"}
