# Stirrup Slack Integration

Run Stirrup agents from Slack via @mentions.

## 1. Create a Slack App

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and click **Create New App** > **From scratch**
2. Name it (e.g. "Stirrup") and select your workspace

### Enable Socket Mode

3. Go to **Settings > Socket Mode** and toggle it **on**
4. Create an app-level token with the `connections:write` scope — name it anything (e.g. "socket")
5. Copy the **App-Level Token** (`xapp-...`)

### Add Bot Scopes

6. Go to **OAuth & Permissions > Bot Token Scopes** and add:
   - `app_mentions:read`
   - `chat:write`
   - `files:read`
   - `files:write`

### Subscribe to Events

7. Go to **Event Subscriptions** and toggle **on**
8. Under **Subscribe to bot events**, add:
   - `app_mention`

### Install to Workspace

9. Go to **Install App** and click **Install to Workspace**
10. Copy the **Bot User OAuth Token** (`xoxb-...`)

### Invite the Bot

11. In Slack, invite the bot to any channel you want it in: `/invite @Stirrup`

## 2. Install

```bash
pip install stirrup[slack,docker]
```

The `docker` extra is needed because the default code execution backend runs in Docker containers. Make sure Docker is running on your machine.

## 3. Run (Quick Start)

Set three environment variables and run:

```bash
export SLACK_BOT_TOKEN=xoxb-...
export SLACK_APP_TOKEN=xapp-...
export OPENROUTER_API_KEY=sk-or-...

python -m stirrup.integrations.slack
```

This starts a bot using `google/gemini-3-flash-preview` via OpenRouter as the default model.

## 4. Run (Custom Agents)

For custom agent configurations, write a Python script:

```python
import asyncio
import os

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.integrations.slack import SlackBot, SlackBotConfig, SlackAgentConfig

client = ChatCompletionsClient(
    model="google/gemini-3-flash-preview",
    base_url="https://openrouter.ai/api/v1",
)

config = SlackBotConfig(
    slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
    slack_app_token=os.environ["SLACK_APP_TOKEN"],
    default_agent=SlackAgentConfig(name="assistant", client=client),
    named_agents={
        "data": SlackAgentConfig(
            name="data-analyst",
            client=client,
            system_prompt="You are a data analysis expert.",
            skills_dir="./skills",  # optional: load SKILL.md files
        ),
    },
)

asyncio.run(SlackBot(config).start())
```

### Browser Agent Example

To add a browser automation agent, install the `browser` extra and pass `BrowserUseToolProvider` in the tools list:

```bash
pip install stirrup[slack,docker,browser]
```

```python
from stirrup.tools.browser_use import BrowserUseToolProvider
from stirrup.tools.code_backends.docker import DockerCodeExecToolProvider
from stirrup.tools.web import WebToolProvider

browser_agent = SlackAgentConfig(
    name="browser-agent",
    client=client,
    system_prompt="You are a browser automation agent. You can navigate websites, interact with pages, and extract information.",
    tools=[
        BrowserUseToolProvider(headless=True),
        DockerCodeExecToolProvider.from_image("python:3.13-slim"),
        WebToolProvider(),
    ],
)

config = SlackBotConfig(
    ...,
    named_agents={
        "data": data_agent,
        "browser": browser_agent,
    },
)
```

Then in Slack: `@Stirrup agent:browser go to example.com and summarise the page`

See [`examples/slack_bot_example.py`](../../../../examples/slack_bot_example.py) for the full example with all agents.

## 5. Usage in Slack

Once the bot is running:

| Message | What happens |
|---|---|
| `@Stirrup create a python script that prints hello world` | Runs the default agent |
| `@Stirrup agent:data summarise this` + attach a CSV | Runs the "data" named agent with the file as input |
| `@Stirrup agent:browser go to example.com and summarise it` | Runs the browser agent to navigate and extract info |
| `@Stirrup model:anthropic/claude-sonnet-4.5 write a poem` | Runs the default agent but overrides the model |
| `@Stirrup agent:data model:openai/gpt-4o analyze trends` + attach files | Named agent + model override + file input |

- **Responses appear in a thread** under your message to keep the channel clean
- **Output files** (charts, CSVs, etc.) are uploaded back to the thread
- **Multiple users** can use the bot simultaneously (concurrency is capped at `max_concurrent_runs`, default 5)

## Configuration Reference

### `SlackBotConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `slack_bot_token` | `str` | required | Bot User OAuth Token (`xoxb-...`) |
| `slack_app_token` | `str` | required | App-Level Token (`xapp-...`) |
| `default_agent` | `SlackAgentConfig` | required | Agent used when no `agent:` prefix is specified |
| `named_agents` | `dict[str, SlackAgentConfig]` | `{}` | Map of name to agent config, invoked via `agent:<name>` |
| `max_concurrent_runs` | `int` | `5` | Max simultaneous agent runs |
| `output_dir` | `str` | `"./slack_output"` | Base directory for agent output files |
| `show_metadata` | `bool` | `True` | Post token usage / model info after each run |

### `SlackAgentConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | required | Agent name (alphanumeric, hyphens, underscores) |
| `client` | `LLMClient` | required | Pre-configured LLM client (e.g. `ChatCompletionsClient`) |
| `tools` | `list` | `None` | Tools for the agent. `None` = Docker code exec + web tools |
| `system_prompt` | `str` | `None` | Custom system prompt |
| `max_turns` | `int` | `30` | Max agent loop iterations |
| `skills_dir` | `str` | `None` | Path to a directory of SKILL.md files |
