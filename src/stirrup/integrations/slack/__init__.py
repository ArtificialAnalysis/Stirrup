"""Stirrup Slack integration.

Run Stirrup agents from Slack via @mentions.

Requires: pip install stirrup[slack]

Quick start:
    export SLACK_BOT_TOKEN=xoxb-...
    export SLACK_APP_TOKEN=xapp-...
    export OPENROUTER_API_KEY=sk-or-...
    python -m stirrup.integrations.slack

Custom setup:
    from stirrup.clients.chat_completions_client import ChatCompletionsClient
    from stirrup.integrations.slack import SlackBot, SlackBotConfig, SlackAgentConfig

    client = ChatCompletionsClient(model="google/gemini-3-flash-preview", base_url="https://openrouter.ai/api/v1")
    config = SlackBotConfig(
        slack_bot_token=os.environ["SLACK_BOT_TOKEN"],
        slack_app_token=os.environ["SLACK_APP_TOKEN"],
        default_agent=SlackAgentConfig(name="assistant", client=client),
    )
    await SlackBot(config).start()
"""

from stirrup.integrations.slack.slack import SlackAgentConfig, SlackBot, SlackBotConfig, SlackLogger

__all__ = [
    "SlackAgentConfig",
    "SlackBot",
    "SlackBotConfig",
    "SlackLogger",
]
