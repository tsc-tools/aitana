import asyncio
import datetime
import os
import sys
import tempfile
from pathlib import Path

from aitana.bot.record_section import record_section
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, LocalShellBackend
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from obspy import UTCDateTime
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])
# Global variable to hold our initialized agent
agent = None
# The skills live at the repo root, so this needs the workspace (editable) install
SKILLS_DIR = Path(__file__).resolve().parents[4] / "skills"
# The bot's own skills, such as bulletin drafting, ship inside the package
BOT_SKILLS_DIR = Path(__file__).resolve().parent / "data"
# The agent's scratch space for scripts and plots, kept outside the repo
WORKDIR = Path(tempfile.mkdtemp(prefix="aitana-bot-"))
# Commands the agent runs see only these variables, not the Slack or API tokens
SHELL_ENV = {
    k: os.environ[k] for k in ("HOME", "HTTPS_PROXY", "HTTP_PROXY") if k in os.environ
}
# The bot's own interpreter comes first, so `python` has the aitana packages
SHELL_ENV["PATH"] = os.pathsep.join(
    [str(Path(sys.executable).parent), os.environ.get("PATH", "")]
)
SYSTEM_PROMPT = (
    "You answer volcano monitoring questions in Slack. Write and run Python "
    "with the aitana packages to fetch data and make plots. Save plots as PNG "
    "files in the working directory; every PNG you write is posted to the "
    "Slack thread with your reply. The conversation is the Slack thread so "
    "far; each message from a person starts with their Slack mention."
)


@tool
def generate_record_section_plot(
    volcano: str, starttime: datetime.datetime, endtime: datetime.datetime
) -> str:
    """Generate a record section plot for the specified volcano and time range."""
    # record_section adds seconds to the start time, which needs a UTCDateTime
    tstart, tend = UTCDateTime(starttime), UTCDateTime(endtime)
    # Written to WORKDIR so the plot is posted to Slack with the reply
    outfile = WORKDIR / f"record_section_{volcano}_{tstart.strftime('%Y%m%dT%H%M')}.png"
    try:
        record_section(volcano, tstart, tend, outfile=str(outfile))
        return str(outfile)
    except Exception as e:
        return f"Record section plot generation failed: {e}"


def _pngs():
    """Map each PNG in the agent's work directory to its modification time.

    Returns
    -------
    dict of pathlib.Path to float
        Modification time of every ``*.png`` under ``WORKDIR``.
    """
    return {p: p.stat().st_mtime for p in WORKDIR.rglob("*.png")}


async def _thread_messages(client, channel, thread_ts, bot_id):
    """Turn a Slack thread into the agent's conversation.

    The bot's own posts become assistant turns. Everyone else's become user
    turns that start with the author's Slack mention, so the agent can tell
    people apart.

    Parameters
    ----------
    client : slack_sdk.web.async_client.AsyncWebClient
        Slack Web API client.
    channel : str
        ID of the channel the thread is in.
    thread_ts : str
        Timestamp of the thread's parent message.
    bot_id : str
        This app's bot ID, which marks the bot's own posts.

    Returns
    -------
    list of dict
        Messages with ``role`` and ``content`` keys, oldest first.
    """
    replies, cursor = [], None
    while True:
        response = await client.conversations_replies(
            channel=channel, ts=thread_ts, cursor=cursor
        )
        replies.extend(response["messages"])
        cursor = response.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    messages = []
    for msg in replies:
        # File uploads, such as the plots the bot posts, carry no text
        if not msg.get("text"):
            continue
        # A thread can open with a post from this app, e.g. a Grafana alarm
        # sent through it; that goes in as context so the conversation opens
        # with a user turn
        if msg.get("bot_id") == bot_id and messages:
            messages.append({"role": "assistant", "content": msg["text"]})
        else:
            author = f"<@{msg['user']}>" if "user" in msg else msg.get("username", "")
            messages.append({"role": "user", "content": f"{author}: {msg['text']}"})
    return messages


async def initialize_bot():
    global agent
    tools = [generate_record_section_plot]

    # Grafana is optional: its tool schemas are sent with every model call, so
    # they cost tokens even when unused
    grafana_mcp_url = os.environ.get("GRAFANA_MCP_URL")
    if grafana_mcp_url:
        # Each MCP tool call opens its own session, so no client has to stay open
        client = MultiServerMCPClient(
            {"grafana": {"url": grafana_mcp_url, "transport": "http"}}
        )
        tools += await client.get_tools()
    # Code runs in WORKDIR; /skills/ maps onto the repo's skills directory and
    # /bot-skills/ onto the package's own
    backend = CompositeBackend(
        default=LocalShellBackend(root_dir=WORKDIR, env=SHELL_ENV),
        routes={
            "/skills/": FilesystemBackend(root_dir=SKILLS_DIR, virtual_mode=True),
            "/bot-skills/": FilesystemBackend(
                root_dir=BOT_SKILLS_DIR, virtual_mode=True
            ),
        },
    )
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-5",
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        skills=["/skills/", "/bot-skills/"],
        backend=backend,
        # The file tools may read the skills but not change them
        permissions=[
            FilesystemPermission(
                operations=["write"],
                paths=["/skills/**", "/bot-skills/**"],
                mode="deny",
            )
        ],
    )


@app.event("app_mention")
async def handle_mention(event, say, client, context):
    # Reply in the thread the mention is in, or start one under it
    thread_ts = event.get("thread_ts", event["ts"])
    before = _pngs()

    # The whole thread is the conversation, so follow-ups can refer back to
    # earlier messages, even those that didn't mention the bot, and nothing
    # is lost when the bot restarts
    messages = await _thread_messages(
        client, event["channel"], thread_ts, context.bot_id
    )
    response = await agent.ainvoke({"messages": messages})

    # .text drops the thinking blocks Claude returns alongside its answer
    await say(text=response["messages"][-1].text, thread_ts=thread_ts)

    # Post every plot the agent wrote or overwrote during this run
    plots = [p for p, mtime in _pngs().items() if before.get(p) != mtime]
    if plots:
        await client.files_upload_v2(
            channel=event["channel"],
            thread_ts=thread_ts,
            file_uploads=[{"file": str(p), "title": p.name} for p in plots],
        )


async def main():
    """Build the agent, then serve Slack events over Socket Mode.

    Both run in one event loop, which the async listeners and the agent share.
    """
    await initialize_bot()
    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    await handler.start_async()


def run():
    """Start the bot; the entry point of the ``aitana-bot`` command."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
