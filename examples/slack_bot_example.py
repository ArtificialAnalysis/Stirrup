"""Example: Stirrup Slack bot with custom agents.

Setup:
    1. Create a Slack App at https://api.slack.com/apps
    2. Enable Socket Mode → generates App-Level Token (xapp-...)
    3. Add Bot Scopes: app_mentions:read, chat:write, files:read, files:write
    4. Subscribe to bot events: app_mention
    5. Install to workspace → generates Bot Token (xoxb-...)
    6. Invite the bot to channels

    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_APP_TOKEN=xapp-...
    export OPENROUTER_API_KEY=sk-or-...
    python examples/slack_bot_example.py

Usage in Slack:
    @stirrup create a python script that prints hello world
    @stirrup agent:data summarize this dataset (+ attach a CSV)
    @stirrup model:anthropic/claude-sonnet-4.5 write a poem
"""

import asyncio
import os

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.integrations.slack import SlackAgentConfig, SlackBot, SlackBotConfig

# Default: OpenRouter with Gemini 3 Flash
client = ChatCompletionsClient(
    model="google/gemini-3-flash-preview",
    base_url="https://openrouter.ai/api/v1",
)

# Default agent — used when no agent: prefix is specified
default = SlackAgentConfig(name="assistant", client=client)

# Named agent — invoked with @stirrup agent:data ...
# skills_dir loads SKILL.md files from a directory and makes them available to the agent.
data_agent = SlackAgentConfig(
    name="data-analyst",
    client=client,
    system_prompt="""
    You are a data analysis expert. Focus on data processing, visualization, and statistical analysis.
    """,
    skills_dir="./skills",  # optional: load skills from this directory
)

config = SlackBotConfig(
    slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
    slack_app_token=os.environ["SLACK_APP_TOKEN"],
    default_agent=default,
    named_agents={
        "data": data_agent,
    },
)

asyncio.run(SlackBot(config).start())
