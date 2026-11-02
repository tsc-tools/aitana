import asyncio
import os

# slack_bot builds its Slack app at import time, which needs a token
os.environ.setdefault("SLACK_BOT_TOKEN", "xoxb-test")

from aitana.bot import slack_bot
from aitana.bot.slack_bot import _thread_messages

BOT_ID = "B0BOT"


class FakeClient:
    """Serve canned pages of ``conversations.replies`` results."""

    def __init__(self, pages):
        self.pages = pages
        self.cursors = []

    async def conversations_replies(self, channel, ts, cursor=None):
        self.cursors.append(cursor)
        return self.pages[len(self.cursors) - 1]


def test_thread_messages_roles_and_paging():
    """People become user turns, the bot's replies assistant turns, across pages."""
    client = FakeClient(
        [
            {
                "messages": [
                    {"user": "U1", "text": "<@U0BOT> draft a VAB for Whakaari"},
                    {"user": "U0BOT", "bot_id": BOT_ID, "text": "DRAFT VAB ..."},
                ],
                "response_metadata": {"next_cursor": "page2"},
            },
            {
                "messages": [
                    # the plot the bot uploaded with its reply
                    {"user": "U0BOT", "bot_id": BOT_ID, "text": ""},
                    {"user": "U2", "text": "The flight was on 31 October"},
                ],
                "response_metadata": {"next_cursor": ""},
            },
        ]
    )
    messages = asyncio.run(_thread_messages(client, "C1", "1.0", BOT_ID))
    assert client.cursors == [None, "page2"]
    assert messages == [
        {"role": "user", "content": "<@U1>: <@U0BOT> draft a VAB for Whakaari"},
        {"role": "assistant", "content": "DRAFT VAB ..."},
        {"role": "user", "content": "<@U2>: The flight was on 31 October"},
    ]


def test_thread_opening_with_app_post_is_user_turn():
    """An alarm the app posted as the thread's parent comes in as context."""
    client = FakeClient(
        [
            {
                "messages": [
                    {"user": "U0BOT", "bot_id": BOT_ID, "text": "RSAM alarm"},
                    {"user": "U1", "text": "<@U0BOT> is this an eruption?"},
                ]
            }
        ]
    )
    messages = asyncio.run(_thread_messages(client, "C1", "1.0", BOT_ID))
    assert [m["role"] for m in messages] == ["user", "user"]
    assert messages[0]["content"] == "<@U0BOT>: RSAM alarm"


def _built_tools(monkeypatch):
    """Run ``initialize_bot`` and return the tools it gave the agent."""
    built = {}
    monkeypatch.setattr(
        slack_bot, "create_deep_agent", lambda **kwargs: built.update(kwargs)
    )
    asyncio.run(slack_bot.initialize_bot())
    return built["tools"]


def test_bot_runs_without_grafana(monkeypatch):
    """Without GRAFANA_MCP_URL the agent gets no MCP tools and no client is made."""
    monkeypatch.delenv("GRAFANA_MCP_URL", raising=False)
    monkeypatch.setattr(slack_bot, "MultiServerMCPClient", None)
    assert _built_tools(monkeypatch) == [slack_bot.generate_record_section_plot]


def test_grafana_tools_added_when_configured(monkeypatch):
    """With GRAFANA_MCP_URL set, the MCP server's tools come after the bot's own."""

    class FakeMCPClient:
        def __init__(self, connections):
            assert connections["grafana"]["url"] == "http://grafana:8000/mcp"

        async def get_tools(self):
            return ["search_dashboards"]

    monkeypatch.setenv("GRAFANA_MCP_URL", "http://grafana:8000/mcp")
    monkeypatch.setattr(slack_bot, "MultiServerMCPClient", FakeMCPClient)
    assert _built_tools(monkeypatch) == [
        slack_bot.generate_record_section_plot,
        "search_dashboards",
    ]
