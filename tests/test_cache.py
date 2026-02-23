"""Tests for CacheManager incremental file sync."""

import os
import shutil
import time
from pathlib import Path

import pytest

from stirrup.core.cache import CacheManager, CacheState, _sync_files_incremental


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache_manager(tmp_path: Path) -> CacheManager:
    """CacheManager backed by a temporary directory."""
    return CacheManager(cache_base_dir=tmp_path / "cache")


@pytest.fixture
def minimal_state() -> CacheState:
    """Minimal valid CacheState for use in save_state calls."""
    return CacheState(
        msgs=[],
        full_msg_history=[],
        turn=0,
        run_metadata={},
        task_hash="aabbccdd1122",
    )


# ---------------------------------------------------------------------------
# Unit tests for _sync_files_incremental
# ---------------------------------------------------------------------------


def test_sync_first_save_copies_all_files(tmp_path: Path) -> None:
    """First sync into an empty dst copies every file including nested ones."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_text("hello")
    (src / "b.txt").write_text("world")
    sub = src / "subdir"
    sub.mkdir()
    (sub / "c.txt").write_text("nested")

    dst = tmp_path / "dst"

    _sync_files_incremental(src, dst)

    assert (dst / "a.txt").read_text() == "hello"
    assert (dst / "b.txt").read_text() == "world"
    assert (dst / "subdir" / "c.txt").read_text() == "nested"


def test_sync_incremental_skips_unchanged_files(tmp_path: Path) -> None:
    """Second sync does not re-copy files whose mtime and size are unchanged."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "stable.txt").write_text("same")
    dst = tmp_path / "dst"

    _sync_files_incremental(src, dst)
    mtime_before = (dst / "stable.txt").stat().st_mtime

    # Run sync again without touching the source file
    _sync_files_incremental(src, dst)

    assert (dst / "stable.txt").stat().st_mtime == mtime_before


def test_sync_copies_modified_files(tmp_path: Path) -> None:
    """A file with a newer mtime in src is re-copied on the next sync."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "file.txt").write_text("original")
    dst = tmp_path / "dst"

    _sync_files_incremental(src, dst)

    # Advance src mtime by 2 seconds to guarantee it is strictly newer than dst
    src_file = src / "file.txt"
    src_file.write_text("modified")
    stat = os.stat(src_file)
    os.utime(src_file, (stat.st_atime, stat.st_mtime + 2))

    _sync_files_incremental(src, dst)

    assert (dst / "file.txt").read_text() == "modified"


def test_sync_deletes_files_removed_from_source(tmp_path: Path) -> None:
    """Files removed from src are deleted from dst on the next sync."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "keep.txt").write_text("keep")
    (src / "gone.txt").write_text("will be removed")
    dst = tmp_path / "dst"

    _sync_files_incremental(src, dst)
    assert (dst / "gone.txt").exists()

    (src / "gone.txt").unlink()
    _sync_files_incremental(src, dst)

    assert not (dst / "gone.txt").exists()
    assert (dst / "keep.txt").exists()


def test_sync_deletes_subdirs_removed_from_source(tmp_path: Path) -> None:
    """Subdirectories removed from src are removed from dst on the next sync."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "root.txt").write_text("root")
    sub = src / "subdir"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested")
    dst = tmp_path / "dst"

    _sync_files_incremental(src, dst)
    assert (dst / "subdir" / "nested.txt").exists()

    shutil.rmtree(sub)
    _sync_files_incremental(src, dst)

    assert not (dst / "subdir").exists()
    assert (dst / "root.txt").exists()


# ---------------------------------------------------------------------------
# Integration tests for CacheManager.save_state
# ---------------------------------------------------------------------------


def test_save_state_creates_files_dir(
    cache_manager: CacheManager,
    tmp_path: Path,
    minimal_state: CacheState,
) -> None:
    """save_state with exec_env_dir creates the files dir and populates it."""
    src = tmp_path / "exec_env"
    src.mkdir()
    (src / "result.py").write_text("print('done')")

    cache_manager.save_state("aabbccdd1122", minimal_state, exec_env_dir=src)

    files_dir = cache_manager._get_files_dir("aabbccdd1122")
    assert files_dir.exists()
    assert (files_dir / "result.py").read_text() == "print('done')"


def test_save_state_no_exec_env_dir(
    cache_manager: CacheManager,
    minimal_state: CacheState,
) -> None:
    """save_state with exec_env_dir=None writes only state.json, no files dir."""
    cache_manager.save_state("aabbccdd1122", minimal_state, exec_env_dir=None)

    assert cache_manager._get_state_file("aabbccdd1122").exists()
    assert not cache_manager._get_files_dir("aabbccdd1122").exists()
