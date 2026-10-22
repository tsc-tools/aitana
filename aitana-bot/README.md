# aitana-bot

Planned interactive duty-officer companion: ask questions about the monitoring data from
Slack, draft bulletins and work through eruption confirmation. The plan is in
[PLAN.md](PLAN.md); the package is currently a stub.

Alarms are not sent by the bot. Workflows such as `ruapehu_mp` store an alarm series in
`aitana-storage`, `aitana-api` serves it, and Grafana alerting posts to Slack when it is raised
(see the root [README](../README.md)).

## Slack setup (once)

The same Slack app can serve the bot and Grafana's Slack contact point, so alarms and
answers come from one bot user.

1. Create an app at <https://api.slack.com/apps> ("From scratch") in your workspace.
2. Under **OAuth & Permissions**, add the bot token scopes `chat:write` and `files:write`.
3. **Install to Workspace** and copy the *Bot User OAuth Token* (`xoxb-…`).
4. In the target channel run `/invite @<app name>`, then copy the channel ID
   (channel name → *About* → bottom of the panel, e.g. `C0123456789`).
5. Provide both through the environment. Never commit them:

   ```bash
   export SLACK_BOT_TOKEN=xoxb-…
   export SLACK_CHANNEL_ID=C0123456789
   ```
