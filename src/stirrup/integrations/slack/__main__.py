"""Run Stirrup Slack bot with default config from environment variables.

Usage:
    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_APP_TOKEN=xapp-...
    export OPENROUTER_API_KEY=sk-or-...
    python -m stirrup.integrations.slack
"""

import asyncio
import os

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.integrations.slack import SlackAgentConfig, SlackBot, SlackBotConfig

client = ChatCompletionsClient(
    model="google/gemini-3-flash-preview",
    base_url="https://openrouter.ai/api/v1",
)

config = SlackBotConfig(
    slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
    slack_app_token=os.environ["SLACK_APP_TOKEN"],
    default_agent=SlackAgentConfig(name="assistant", client=client),
)

asyncio.run(SlackBot(config).start())
