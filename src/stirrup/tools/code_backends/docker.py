"""Docker container execution environment backend for code execution."""

import contextlib
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Self

from anyio import fail_after, to_thread
from dotenv import load_dotenv

from stirrup.core.models import ImageContentBlock

try:
    import docker
    from docker.client import DockerClient
    from docker.errors import APIError, BuildError, ImageNotFound, NotFound
    from docker.models.containers import Container
except ImportError as e:
    raise ImportError(
        "Requires installation of the docker extra. Install with (for example): `uv pip install stirrup[docker]` or `uv add stirrup[docker]`",
    ) from e

import logging

from stirrup.core.models import Tool, ToolUseCountMetadata

from .base import (
    SHELL_TIMEOUT,
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
    SavedFile,
    SaveOutputFilesResult,
    UploadedFile,
    UploadFilesResult,
)

logger = logging.getLogger(__name__)

DEFAULT_WORKING_DIR = "/workspace"


class DockerCodeExecToolProvider(CodeExecToolProvider):
    """Docker container code execution tool provider.

    Creates a persistent Docker container with a host directory mounted
    as a volume. Commands are executed via docker exec, and files persist
    between commands within the same session.

    Usage:
        # From pre-built image
        provider = DockerCodeExecToolProvider.from_image("python:3.12-slim")

        # From Dockerfile
        provider = DockerCodeExecToolProvider.from_dockerfile(Path("./Dockerfile"))

        # With command allowlist
        provider = DockerCodeExecToolProvider.from_image(
            "python:3.12-slim",
            allowed_commands=[r"^python", r"^pip"],
        )

        # With Agent
        from stirrup.clients.chat_completions_client import ChatCompletionsClient

        client = ChatCompletionsClient(model="gpt-5")
        agent = Agent(client=client, name="assistant", tools=[provider])
        async with agent.session() as session:
            await session.run("Run Python code")
    """

    def __init__(
        self,
        source: str | Path,
        *,
        is_dockerfile: bool = False,
        dockerfile_context: Path | None = None,
        working_dir: str = DEFAULT_WORKING_DIR,
        allowed_commands: list[str] | None = None,
        temp_base_dir: Path | None = None,
        env_vars: list[str] | None = None,
    ) -> None:
        """Initialize DockerCodeExecToolProvider configuration.

        Args:
            source: Docker image name (e.g., "python:3.12-slim") or path to Dockerfile.
            is_dockerfile: If True, source is treated as a Dockerfile path. Default False.
            dockerfile_context: Build context directory for Dockerfile builds.
            working_dir: Container working directory (default: /workspace).
            allowed_commands: Optional regex patterns for command allowlist.
            temp_base_dir: Optional host base directory for temp files.
            env_vars: Optional list of environment variable names to inject into the
                container. Values are loaded from the current environment (os.environ)
                after calling load_dotenv() to load any .env file.

        Prefer using the factory methods for clarity:
        - DockerCodeExecToolProvider.from_image() for pre-built images
        - DockerCodeExecToolProvider.from_dockerfile() for building from Dockerfile

        """
        super().__init__(allowed_commands=allowed_commands)

        self._source = source
        self._is_dockerfile = is_dockerfile
        self._dockerfile_context = dockerfile_context
        self._working_dir = working_dir
        self._temp_base_dir = temp_base_dir
        self._env_vars = env_vars

        # Runtime state
        self._temp_dir: Path | None = None
        self._client: DockerClient | None = None
        self._container: Container | None = None

    @property
    def temp_dir(self) -> Path | None:
        """Return the host temp directory path, or None if not started."""
        return self._temp_dir

    @property
    def container_id(self) -> str | None:
        """Return the container short ID, or None if not started."""
        return self._container.short_id if self._container else None

    def _resolve_file_path(self, path: str) -> Path:
        """Resolve a container path string to a validated host file path.

        Args:
            path: Path to file (relative to working directory, or absolute container path).

        Returns:
            Resolved absolute host Path to the file.

        Raises:
            RuntimeError: If execution environment not started.
            ValueError: If path is outside mounted directory or is not a file.
            FileNotFoundError: If file does not exist.

        """
        if self._temp_dir is None:
            raise RuntimeError("ExecutionEnvironment not started. Use 'async with exec_env.create()' first.")

        file_path = Path(path)

        # Handle both absolute container paths and relative paths
        if file_path.is_absolute():
            # Convert container absolute path to host path
            # e.g., /workspace/image.png -> <temp_dir>/image.png
            if str(file_path).startswith(self._working_dir):
                relative = file_path.relative_to(self._working_dir)
                file_path = self._temp_dir / relative
            else:
                raise ValueError(f"Path is outside mounted directory: {path}")
        else:
            file_path = self._temp_dir / file_path

        # Security check: ensure path is within temp directory
        try:
            file_path.resolve().relative_to(self._temp_dir.resolve())
        except ValueError:
            raise ValueError(f"Path is outside execution environment directory: {path}") from None

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        return file_path

    @classmethod
    def from_image(
        cls,
        image: str,
        *,
        working_dir: str = DEFAULT_WORKING_DIR,
        allowed_commands: list[str] | None = None,
        temp_base_dir: Path | str | None = None,
        env_vars: list[str] | None = None,
    ) -> Self:
        """Create tool provider from a pre-built Docker image.

        Args:
            image: Docker image name (e.g., "python:3.12-slim").
            working_dir: Container working directory (default: /workspace).
            allowed_commands: Optional regex patterns for command allowlist.
            temp_base_dir: Optional host base directory for temp files.
            env_vars: Optional list of environment variable names to inject into the
                container. Values are loaded from os.environ (after load_dotenv()).

        Returns:
            Configured DockerCodeExecToolProvider instance.

        Example:
            provider = DockerCodeExecToolProvider.from_image(
                "python:3.12-slim",
                env_vars=["OPENROUTER_API_KEY", "DATABASE_URL"],
            )
            async with provider as tool:
                result = await provider.run_command("python --version")

        """
        return cls(
            image,
            is_dockerfile=False,
            working_dir=working_dir,
            allowed_commands=allowed_commands,
            temp_base_dir=Path(temp_base_dir) if temp_base_dir else None,
            env_vars=env_vars,
        )

    @classmethod
    def from_dockerfile(
        cls,
        dockerfile: Path | str,
        *,
        context: Path | str | None = None,
        working_dir: str = DEFAULT_WORKING_DIR,
        allowed_commands: list[str] | None = None,
        temp_base_dir: Path | str | None = None,
        env_vars: list[str] | None = None,
    ) -> Self:
        """Create tool provider by building from a Dockerfile.

        Args:
            dockerfile: Path to the Dockerfile.
            context: Build context directory. Defaults to Dockerfile's parent.
            working_dir: Container working directory (default: /workspace).
            allowed_commands: Optional regex patterns for command allowlist.
            temp_base_dir: Optional host base directory for temp files.
            env_vars: Optional list of environment variable names to inject into the
                container. Values are loaded from os.environ (after load_dotenv()).

        Returns:
            Configured DockerCodeExecToolProvider instance.

        Example:
            provider = DockerCodeExecToolProvider.from_dockerfile(
                Path("./Dockerfile"),
                env_vars=["OPENROUTER_API_KEY", "ANTHROPIC_API_KEY"],
            )
            async with provider as tool:
                result = await provider.run_command("python script.py")

        """
        return cls(
            dockerfile,
            is_dockerfile=True,
            dockerfile_context=Path(context) if context else None,
            working_dir=working_dir,
            allowed_commands=allowed_commands,
            temp_base_dir=Path(temp_base_dir) if temp_base_dir else None,
            env_vars=env_vars,
        )

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        """Initialize Docker container and return the code_exec tool.

        Creates a temp directory on the host, initializes the Docker client,
        prepares the image (pull or build), and starts a persistent container
        with the temp directory mounted as a volume.
        """
        # 1. Load environment variables from .env file
        load_dotenv()

        # 2. Build environment dict from requested env var names
        env_dict: dict[str, str] = {}
        if self._env_vars:
            for name in self._env_vars:
                if name in os.environ:
                    env_dict[name] = os.environ[name]
                else:
                    logger.warning("Requested env var '%s' not found in environment", name)
            if env_dict:
                logger.debug("Injecting environment variables: %s", list(env_dict.keys()))

        # 3. Create temp directory on host
        if self._temp_base_dir:
            self._temp_base_dir.mkdir(parents=True, exist_ok=True)
        self._temp_dir = Path(tempfile.mkdtemp(prefix="docker_exec_env_", dir=self._temp_base_dir))

        # 4. Initialize Docker client
        self._client = await to_thread.run_sync(docker.from_env)
        if self._client is None:
            raise RuntimeError("Failed to connect to Docker daemon. Is Docker running?")
        client = self._client  # Capture for lambda type narrowing

        # 5. Prepare image (pull or build)
        image_name = await self._prepare_image()

        # 6. Start container with volume mount and environment variables
        self._container = await to_thread.run_sync(
            lambda: client.containers.run(
                image_name,
                command="tail -f /dev/null",  # Keep container running
                detach=True,
                volumes={
                    str(self._temp_dir): {
                        "bind": self._working_dir,
                        "mode": "rw",
                    },
                },
                working_dir=self._working_dir,
                environment=env_dict if env_dict else None,
                remove=False,  # We handle removal manually
            )
        )
        logger.info("Started container: %s (image: %s)", self._container.short_id, image_name)
        return self.get_code_exec_tool()

    async def _prepare_image(self) -> str:
        """Prepare Docker image (pull pre-built or build from Dockerfile).

        Returns:
            The image name/tag to use for container creation.

        Raises:
            RuntimeError: If image build or pull fails.

        """
        if self._client is None:
            raise RuntimeError("Docker client not initialized")
        client = self._client  # Capture for lambda type narrowing

        if self._is_dockerfile:
            # Build from Dockerfile
            dockerfile_path = Path(self._source).resolve()
            context_path = self._dockerfile_context.resolve() if self._dockerfile_context else dockerfile_path.parent

            # Generate unique tag based on dockerfile path
            tag = f"stirrup-exec-env-{hashlib.md5(str(dockerfile_path).encode()).hexdigest()[:8]}"

            logger.info("Building image from %s with tag %s", dockerfile_path, tag)

            try:
                # Determine dockerfile path relative to context
                if dockerfile_path.is_relative_to(context_path):
                    dockerfile_rel = str(dockerfile_path.relative_to(context_path))
                else:
                    dockerfile_rel = str(dockerfile_path)

                _image, build_logs = await to_thread.run_sync(
                    lambda: client.images.build(
                        path=str(context_path),
                        dockerfile=dockerfile_rel,
                        tag=tag,
                        rm=True,  # Remove intermediate containers
                    )
                )
                for log in build_logs:
                    if "stream" in log:
                        logger.debug("Build: %s", log["stream"].strip())
                return tag
            except BuildError as exc:
                raise RuntimeError(f"Failed to build Docker image: {exc}") from exc
        else:
            # Pull pre-built image
            image_name = str(self._source)

            try:
                # Check if image exists locally
                await to_thread.run_sync(client.images.get, image_name)
                logger.debug("Image %s found locally", image_name)
            except ImageNotFound:
                logger.info("Pulling image: %s", image_name)
                try:
                    await to_thread.run_sync(client.images.pull, image_name)
                except APIError as exc:
                    raise RuntimeError(f"Failed to pull Docker image '{image_name}': {exc}") from exc

            return image_name

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Stop container and cleanup temp directory."""
        # Stop and remove container
        if self._container:
            container = self._container  # Capture for lambda type narrowing
            try:
                logger.info("Stopping container: %s", container.short_id)
                await to_thread.run_sync(lambda: container.stop(timeout=10))
                await to_thread.run_sync(lambda: container.remove(force=True))
                logger.info("Removed container: %s", container.short_id)
            except NotFound:
                logger.debug("Container already removed")
            except Exception as exc:
                logger.warning("Failed to cleanup container: %s", exc)
            self._container = None

        # Close Docker client
        if self._client:
            with contextlib.suppress(Exception):
                await to_thread.run_sync(self._client.close)
            self._client = None

        # Cleanup temp directory
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as exc:
                logger.warning("Failed to cleanup temp directory %s: %s", self._temp_dir, exc)
        self._temp_dir = None

    def _container_path_to_host(self, path: str) -> Path:
        """Convert a container path to the corresponding host path.

        Args:
            path: Path in the container (relative or absolute).

        Returns:
            Resolved Path on the host filesystem.

        Raises:
            RuntimeError: If environment not started.
            ValueError: If path is outside the mounted directory.

        """
        if self._temp_dir is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        source_path = Path(path)

        # Handle both absolute container paths and relative paths
        if source_path.is_absolute():
            # Convert container absolute path to host path
            # e.g., /workspace/output.txt -> <temp_dir>/output.txt
            if str(source_path).startswith(self._working_dir):
                relative = source_path.relative_to(self._working_dir)
                host_path = self._temp_dir / relative
            else:
                raise ValueError(f"Path is outside mounted directory: {path}")
        else:
            host_path = self._temp_dir / source_path

        # Security: ensure path is within temp directory
        try:
            host_path.resolve().relative_to(self._temp_dir.resolve())
        except ValueError as e:
            raise ValueError(f"Path is outside execution environment: {path}") from e

        return host_path

    async def read_file_bytes(self, path: str) -> bytes:
        """Read file content as bytes from the container.

        Since files are volume-mounted, reads directly from the host temp directory.

        Args:
            path: File path (relative or absolute container path).

        Returns:
            File contents as bytes.

        Raises:
            RuntimeError: If environment not started.
            ValueError: If path is outside mounted directory.
            FileNotFoundError: If file does not exist.

        """
        host_path = self._container_path_to_host(path)
        if not host_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return host_path.read_bytes()

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        """Write bytes to a file in the container.

        Since files are volume-mounted, writes directly to the host temp directory.

        Args:
            path: Destination path (relative or absolute container path).
            content: File contents to write.

        Raises:
            RuntimeError: If environment not started.
            ValueError: If path is outside mounted directory.

        """
        host_path = self._container_path_to_host(path)
        host_path.parent.mkdir(parents=True, exist_ok=True)
        host_path.write_bytes(content)

    async def run_command(self, cmd: str, *, timeout: int = SHELL_TIMEOUT) -> CommandResult:
        """Execute a shell command in the Docker container.

        Args:
            cmd: Shell command to execute (bash syntax).
            timeout: Maximum time in seconds to wait for command completion.

        Returns:
            CommandResult with exit_code, stdout, stderr, and optional error info.

        """
        if self._container is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )
        container = self._container  # Capture for lambda type narrowing

        # Check allowlist
        if not self._check_allowed(cmd):
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: '{cmd}' does not match any allowed patterns",
                error_kind="command_not_allowed",
                advice="Only commands matching the allowlist patterns are permitted.",
            )

        try:
            # Execute command with timeout
            with fail_after(timeout):
                exec_result = await to_thread.run_sync(
                    lambda: container.exec_run(
                        cmd=["bash", "-c", cmd],
                        workdir=self._working_dir,
                        demux=True,  # Separate stdout/stderr
                    )
                )

            exit_code = exec_result.exit_code
            stdout_bytes, stderr_bytes = exec_result.output

            return CommandResult(
                exit_code=exit_code,
                stdout=(stdout_bytes or b"").decode("utf-8", errors="replace"),
                stderr=(stderr_bytes or b"").decode("utf-8", errors="replace"),
            )

        except TimeoutError:
            logger.warning("Command timed out after %d seconds: %s", timeout, cmd[:100])
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                error_kind="timeout",
            )
        except APIError as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="docker_api_error",
                advice="Docker API error occurred. Check Docker daemon is running.",
            )
        except Exception as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="execution_error",
            )

    async def save_output_files(
        self,
        paths: list[str],
        output_dir: Path | str,
        dest_env: "CodeExecToolProvider | None" = None,
    ) -> SaveOutputFilesResult:
        """Move files from the mounted temp directory to a destination.

        Since files are volume-mounted, they're already on the host.

        When dest_env is None (local filesystem), files are MOVED (not copied) -
        originals are deleted from the execution environment.
        Existing files in output_dir are silently overwritten.

        When dest_env is provided (cross-environment transfer), files are copied
        using the base class implementation via read/write primitives.

        Args:
            paths: List of file paths in the execution environment (relative or absolute container paths).
                   Relative paths are resolved against the container working directory.
                   Absolute container paths starting with working_dir are mapped to the host.
            output_dir: Directory path to save files to.
            dest_env: If provided, output_dir is interpreted as a path within dest_env
                      (cross-environment transfer). If None, output_dir is a local
                      filesystem path.

        Returns:
            SaveOutputFilesResult containing lists of saved files and any failures.

        """
        if self._temp_dir is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )

        # If dest_env is provided, use the base class implementation (cross-env transfer)
        if dest_env is not None:
            return await super().save_output_files(paths, output_dir, dest_env)

        # Local filesystem - use optimized move operation
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        result = SaveOutputFilesResult()

        for source_path_str in paths:
            try:
                host_path = self._container_path_to_host(source_path_str)

                if not host_path.exists():
                    result.failed[source_path_str] = "File does not exist"
                    logger.warning("Execution environment file does not exist: %s", source_path_str)
                    continue

                if not host_path.is_file():
                    result.failed[source_path_str] = "Path is not a file"
                    logger.warning("Execution environment path is not a file: %s", source_path_str)
                    continue

                file_size = host_path.stat().st_size
                dest_path = output_dir_path / host_path.name

                # Move file (overwrites if exists)
                shutil.move(str(host_path), str(dest_path))

                result.saved.append(
                    SavedFile(
                        source_path=source_path_str,
                        output_path=dest_path,
                        size=file_size,
                    ),
                )

            except ValueError as exc:
                # Path validation error from _container_path_to_host
                result.failed[source_path_str] = str(exc)
                logger.warning("Path validation error: %s", exc)
            except Exception as exc:
                result.failed[source_path_str] = str(exc)
                logger.exception("Failed to move file: %s", source_path_str)

        return result

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: "CodeExecToolProvider | None" = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        """Upload files to the execution environment.

        Since files are volume-mounted, this copies files to the host temp directory
        which makes them automatically visible in the container at working_dir.

        When source_env is None (local filesystem), files are COPIED (not moved) -
        originals remain on the local filesystem.
        Directories are uploaded recursively, preserving their structure.

        When source_env is provided (cross-environment transfer), files are copied
        using the base class implementation via read/write primitives.

        Args:
            *paths: File or directory paths to upload. If source_env is None, these
                    are local filesystem paths. If source_env is provided, these are
                    paths within source_env.
            source_env: If provided, paths are within source_env. If None, paths are
                        local filesystem paths.
            dest_dir: Destination subdirectory within the container working directory.
                      If None, files are placed directly in the working directory.

        Returns:
            UploadFilesResult containing lists of uploaded files and any failures.

        """
        if self._temp_dir is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )

        # If source_env is provided, use the base class implementation (cross-env transfer)
        if source_env is not None:
            return await super().upload_files(*paths, source_env=source_env, dest_dir=dest_dir)

        # Local filesystem - use optimized copy operation
        dest_base = self._temp_dir / dest_dir if dest_dir else self._temp_dir
        dest_base.mkdir(parents=True, exist_ok=True)

        result = UploadFilesResult()

        for source in paths:
            source = Path(source).resolve()

            if not source.exists():
                result.failed[str(source)] = "File or directory does not exist"
                logger.warning("Upload source does not exist: %s", source)
                continue

            try:
                if source.is_file():
                    dest = dest_base / source.name
                    shutil.copy2(source, dest)
                    # Report path as it appears in container
                    container_path = f"{self._working_dir}/{dest.relative_to(self._temp_dir)}"
                    result.uploaded.append(
                        UploadedFile(
                            source_path=source,
                            dest_path=container_path,
                            size=source.stat().st_size,
                        ),
                    )
                    logger.debug("Uploaded file: %s -> %s", source, container_path)

                elif source.is_dir():
                    # If dest_dir was explicitly provided, copy contents directly to dest_base
                    # Otherwise, create a subdirectory with the source's name
                    if dest_dir:
                        dest = dest_base
                        # Copy contents of source directory into dest_base
                        for item in source.iterdir():
                            item_dest = dest / item.name
                            if item.is_file():
                                shutil.copy2(item, item_dest)
                            else:
                                shutil.copytree(item, item_dest, dirs_exist_ok=True)
                    else:
                        dest = dest_base / source.name
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                    # Track all individual files uploaded
                    for file_path in source.rglob("*"):
                        if file_path.is_file():
                            relative = file_path.relative_to(source)
                            dest_file = dest / relative
                            container_path = f"{self._working_dir}/{dest_file.relative_to(self._temp_dir)}"
                            result.uploaded.append(
                                UploadedFile(
                                    source_path=file_path,
                                    dest_path=container_path,
                                    size=file_path.stat().st_size,
                                ),
                            )
                    logger.debug("Uploaded directory: %s -> %s", source, dest)

            except Exception as exc:
                result.failed[str(source)] = str(exc)
                logger.exception("Failed to upload: %s", source)

        return result

    async def view_image(self, path: str) -> ImageContentBlock:
        """Read and return an image file from the Docker execution environment.

        Args:
            path: Path to image file (relative to working directory, or absolute container path).

        Returns:
            ImageContentBlock containing the image data.

        Raises:
            RuntimeError: If execution environment not started.
            FileNotFoundError: If file does not exist.
            ValueError: If path is outside mounted directory, is a directory, or not a valid image.

        """
        file_bytes = await self.read_file_bytes(path)
        return ImageContentBlock(data=file_bytes)
