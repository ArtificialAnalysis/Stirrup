"""Code execution backends.

This module provides code execution backends for the Agent.

Available here (no optional dependencies):
- Base classes and utilities from .base
- LocalCodeExecToolProvider (uses subprocess)

Optional backends require explicit imports:
- DockerCodeExecToolProvider: `from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider`
- E2BCodeExecToolProvider: `from stirrup.tools.code_backends.e2b import E2BCodeExecToolProvider`
"""

from .base import (
    SHELL_TIMEOUT,
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
    SavedFile,
    SaveOutputFilesResult,
    UploadedFile,
    UploadFilesResult,
    format_result,
)
from .local import LocalCodeExecToolProvider

__all__ = [
    "SHELL_TIMEOUT",
    "CodeExecToolProvider",
    "CodeExecutionParams",
    "CommandResult",
    "LocalCodeExecToolProvider",
    "SaveOutputFilesResult",
    "SavedFile",
    "UploadFilesResult",
    "UploadedFile",
    "format_result",
]
