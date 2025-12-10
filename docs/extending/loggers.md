# Custom Loggers

This guide covers implementing custom loggers for Stirrup.

## AgentLoggerBase

All loggers must implement the [`AgentLoggerBase`][stirrup.utils.logging.AgentLoggerBase] abstract class. Key methods to implement:

| Method | Purpose |
|--------|---------|
| `__enter__` | Called when agent session starts |
| `__exit__` | Called when agent session ends |
| `on_step` | Called after each agent step with progress stats |
| `assistant_message` | Log LLM responses |
| `user_message` | Log user inputs |
| `task_message` | Log initial task/prompt |
| `tool_result` | Log tool execution results |
| `debug`, `info`, `warning`, `error` | Standard logging methods |

Properties set by Agent before `__enter__`: `name`, `model`, `max_turns`, `depth`

Properties set before `__exit__`: `finish_params`, `run_metadata`, `output_dir`

## Minimal Implementation

```python
from stirrup.utils.logging import AgentLoggerBase


class MinimalLogger(AgentLoggerBase):
    """Minimal logger that prints step summaries."""

    def __enter__(self):
        print(f"Starting agent: {self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            print(f"Agent failed: {exc_val}")
        else:
            print(f"Agent completed: {self.finish_params}")

    def on_step(self, turn, total_tools, input_tokens, output_tokens):
        print(f"Turn {turn}: {total_tools} tools, {input_tokens + output_tokens} tokens")
```

## File Logger

```python
import json
from pathlib import Path
from datetime import datetime


class FileLogger(AgentLoggerBase):
    """Logger that writes to a JSON file."""

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_file: Path | None = None
        self.steps: list[dict] = []

    def __enter__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{self.name}_{timestamp}.json"
        self.steps = []

        # Write initial entry
        self._write({
            "event": "start",
            "agent": self.name,
            "model": self.model,
            "max_turns": self.max_turns,
            "timestamp": datetime.now().isoformat(),
        })

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Write final entry
        self._write({
            "event": "end",
            "success": exc_type is None,
            "error": str(exc_val) if exc_val else None,
            "finish_params": self.finish_params.model_dump() if self.finish_params else None,
            "steps": self.steps,
            "timestamp": datetime.now().isoformat(),
        })

    def on_step(self, turn, total_tools, input_tokens, output_tokens):
        step = {
            "turn": turn,
            "total_tools": total_tools,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": datetime.now().isoformat(),
        }
        self.steps.append(step)

    def _write(self, data: dict):
        if self.log_file:
            with open(self.log_file, "w") as f:
                json.dump(data, f, indent=2)
```

## Metrics Logger

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunMetrics:
    """Aggregated metrics for a run."""
    total_turns: int = 0
    total_tools: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    start_time: float = 0
    end_time: float = 0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


class MetricsLogger(AgentLoggerBase):
    """Logger that collects metrics for analysis."""

    def __init__(self):
        self.metrics: RunMetrics | None = None
        self._runs: list[RunMetrics] = []

    def __enter__(self):
        import time
        self.metrics = RunMetrics(start_time=time.time())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        if self.metrics:
            self.metrics.end_time = time.time()
            self._runs.append(self.metrics)

    def on_step(self, turn, total_tools, input_tokens, output_tokens):
        if self.metrics:
            self.metrics.total_turns = turn
            self.metrics.total_tools += total_tools
            self.metrics.total_input_tokens += input_tokens
            self.metrics.total_output_tokens += output_tokens

    def get_all_runs(self) -> list[RunMetrics]:
        """Get metrics from all runs."""
        return self._runs.copy()

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics across all runs."""
        if not self._runs:
            return {}

        return {
            "total_runs": len(self._runs),
            "avg_turns": sum(r.total_turns for r in self._runs) / len(self._runs),
            "avg_tokens": sum(r.total_tokens for r in self._runs) / len(self._runs),
            "avg_duration": sum(r.duration_seconds for r in self._runs) / len(self._runs),
        }
```

## Callback Logger

```python
from collections.abc import Callable


class CallbackLogger(AgentLoggerBase):
    """Logger that invokes callbacks on events."""

    def __init__(
        self,
        on_start: Callable[[str], None] | None = None,
        on_end: Callable[[bool, Any], None] | None = None,
        on_step_callback: Callable[[int, int, int, int], None] | None = None,
    ):
        self._on_start = on_start
        self._on_end = on_end
        self._on_step_callback = on_step_callback

    def __enter__(self):
        if self._on_start:
            self._on_start(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._on_end:
            self._on_end(exc_type is None, self.finish_params)

    def on_step(self, turn, total_tools, input_tokens, output_tokens):
        if self._on_step_callback:
            self._on_step_callback(turn, total_tools, input_tokens, output_tokens)
```

## Using Custom Loggers

```python
from stirrup import Agent
from stirrup.clients.chat_completions_client import ChatCompletionsClient

# With MetricsLogger
metrics_logger = MetricsLogger()
client = ChatCompletionsClient(model="gpt-5")

agent = Agent(
    client=client,
    name="my_agent",
    logger=metrics_logger,
)

async with agent.session() as session:
    await session.run("Do something")

# Access metrics after run
print(metrics_logger.get_summary())
```

## Combining Loggers

```python
class CompositeLogger(AgentLoggerBase):
    """Logger that delegates to multiple loggers."""

    def __init__(self, *loggers: AgentLoggerBase):
        self._loggers = loggers

    def __enter__(self):
        for logger in self._loggers:
            # Copy properties to child loggers
            logger.name = self.name
            logger.model = self.model
            logger.max_turns = self.max_turns
            logger.depth = self.depth
            logger.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for logger in self._loggers:
            logger.finish_params = self.finish_params
            logger.run_metadata = self.run_metadata
            logger.__exit__(exc_type, exc_val, exc_tb)

    def on_step(self, turn, total_tools, input_tokens, output_tokens):
        for logger in self._loggers:
            logger.on_step(turn, total_tools, input_tokens, output_tokens)


# Use multiple loggers
client = ChatCompletionsClient(model="gpt-5")
agent = Agent(
    client=client,
    name="my_agent",
    logger=CompositeLogger(
        FileLogger("./logs"),
        MetricsLogger(),
    ),
)
```

## Sub-Agent Awareness

The `depth` property indicates nesting level:

```python
class IndentedLogger(AgentLoggerBase):
    """Logger that indents output based on sub-agent depth."""

    def __enter__(self):
        indent = "  " * self.depth
        print(f"{indent}Starting: {self.name} (depth={self.depth})")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        indent = "  " * self.depth
        print(f"{indent}Finished: {self.name}")

    def on_step(self, turn, total_tools, input_tokens, output_tokens):
        indent = "  " * self.depth
        print(f"{indent}  Turn {turn}")
```

## Next Steps

- [Custom Backends](code_backends.md) - Code execution backends
- [Custom Clients](clients.md) - LLM client customization
