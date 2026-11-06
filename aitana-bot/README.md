# aitana-bot

Interactive duty-officer companion in Slack. Mention the bot in a channel or thread to ask
about the monitoring data, which it fetches and plots by writing and running Python with the
aitana packages and by querying Grafana, or to draft a Volcanic Activity Bulletin together
with it (the `vab-draft` skill in `src/aitana/bot/data/`). It reads the whole thread, so
follow-ups can refer to earlier messages. The plan, including eruption confirmation, is in
[PLAN.md](PLAN.md).

Alarms are not sent by the bot. Workflows such as `ruapehu_mp` store an alarm series in
`aitana-storage`, `aitana-storage-api` serves it, and Grafana alerting posts to Slack when it is raised
(see the root [README](../README.md)).

## Using the bot

The bot only answers messages that mention it, so start every message to it with
`@aitana-bot` (or whatever you named the app), including follow-ups in a thread. A message
without the mention gets no reply. The bot still reads it as part of the thread the next time
someone mentions it, so you can post the details first and then ask, e.g.
`@aitana-bot draft a VAB from the above`.

## Slack setup (once)

The same Slack app can serve the bot and Grafana's Slack contact point, so alarms and
answers come from one bot user.

1. Create an app at <https://api.slack.com/apps> ("From scratch") in your workspace.
2. Under **OAuth & Permissions**, add the bot token scopes `chat:write` and `files:write`.
   The bot reads the whole thread it is mentioned in, which also needs `channels:history`
   (and `groups:history` for private channels).
3. Under **Socket Mode**, turn it on and create an app-level token with the
   `connections:write` scope (`xapp-…`). The bot connects to Slack through it, so it needs
   no public URL.
4. Under **Event Subscriptions**, turn events on and subscribe to the bot event
   `app_mention`, which adds the `app_mentions:read` scope.
5. **Install to Workspace** and copy the *Bot User OAuth Token* (`xoxb-…`). Reinstall
   whenever you change scopes or events later.
6. In the target channel run `/invite @<app name>`, then copy the channel ID
   (channel name → *About* → bottom of the panel, e.g. `C0123456789`).
7. Provide them through the environment. Never commit them:

   ```bash
   export SLACK_BOT_TOKEN=xoxb-…
   export SLACK_APP_TOKEN=xapp-…
   export SLACK_CHANNEL_ID=C0123456789
   ```

## Anthropic API key

The bot's agent runs on Claude (`anthropic:claude-sonnet-5`, set in `slack_bot.py`) and calls
the Anthropic API with an API key.

1. Create a key in the [Claude Console](https://platform.claude.com) under **Settings → API
   keys**. Use a key just for the bot, so it can be revoked without affecting anything else.
2. Provide it through the environment, like the Slack tokens. Never commit it:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-…
   ```

The model client reads `ANTHROPIC_API_KEY` from the environment. Commands the agent writes
and runs don't get it (see `SHELL_ENV` in `slack_bot.py`). `scripts/run_bot_sandboxed.sh`
refuses to start without it and passes it into the container.

## Grafana MCP server (optional)

The bot can query Grafana through an MCP server, such as
[mcp-grafana](https://github.com/grafana/mcp-grafana), over streamable HTTP. It is off unless
you point the bot at the server's URL:

```bash
export GRAFANA_MCP_URL=http://<host>:<port>/mcp
```

Leave it off unless you need it: every tool the server offers is described in every request
to Claude, whether the bot uses it or not. mcp-grafana's 79 tools add roughly 30–40K tokens
to each request.

The bot sends no credentials, so the MCP server needs its own access to Grafana. In the
sandbox, `docker/init-firewall.sh` lets the bot reach this host and port and nothing else
outside its allowlist.
