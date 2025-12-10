# Full Customization

This guide covers cloning and importing Stirrup locally for deep customization of the framework internals.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/ArtificialAnalysis/Stirrup.git
cd stirrup

# Install in editable mode
pip install -e .      # or: uv venv && uv pip install -e .

# Or with all optional dependencies
pip install -e '.[all]'  # or: uv venv && uv pip install -e '.[all]'
```

## Importing in Your Project

After editable installation, import as usual:

```python
from stirrup import Agent, DEFAULT_TOOLS
from stirrup.core.models import SystemMessage, UserMessage
from stirrup.clients import ChatCompletionsClient
```

## Project Structure

```
src/stirrup/
├── clients/          # LLM client implementations
├── core/             # Agent class, models, exceptions
├── tools/            # Tool implementations
│   └── code_backends/  # Code execution backends
├── utils/            # Logging and text utilities
└── prompts/          # System prompt templates
```

**Common customization points:**

| Directory | Use Case |
|-----------|----------|
| `clients/` | Custom LLM providers or API modifications |
| `core/agent.py` | Agent loop behavior changes |
| `tools/` | New tools or modifying existing ones |
| `tools/code_backends/` | Custom execution environments |

## Next Steps

- [Custom Clients](clients.md) - Implementing LLM client protocol
- [Custom Tools](tools.md) - Advanced tool patterns
- [Custom Backends](code_backends.md) - Execution environment customization
