from pathlib import Path
from typing import Literal

import pytest


@pytest.fixture
def anyio_backend() -> Literal["asyncio"]:
    return "asyncio"


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample file for upload tests."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello, World!")
    return file_path


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a sample directory with files for upload tests."""
    dir_path = tmp_path / "sample_dir"
    dir_path.mkdir()
    (dir_path / "file1.txt").write_text("File 1")
    (dir_path / "file2.txt").write_text("File 2")
    subdir = dir_path / "subdir"
    subdir.mkdir()
    (subdir / "file3.txt").write_text("File 3")
    return dir_path
