# aitana-bot

Posts alerts written by Aitana workflows to Slack. It reads workflow outputs and imports no
other aitana package. The longer-term plan (bulletin drafting, eruption confirmation) is in
[PLAN.md](PLAN.md).

## Slack setup (once)

1. Create an app at <https://api.slack.com/apps> ("From scratch") in your workspace.
2. Under **OAuth & Permissions**, add the bot token scopes `chat:write` and `files:write`.
3. **Install to Workspace** and copy the *Bot User OAuth Token* (`xoxb-…`).
4. In the target channel run `/invite @<app name>`, then copy the channel ID
   (channel name → *About* → bottom of the panel, e.g. `C0123456789`).
5. Provide both to the bot through the environment. Never commit them:

   ```bash
   export SLACK_BOT_TOKEN=xoxb-…
   export SLACK_CHANNEL_ID=C0123456789
   ```

## Usage

```bash
volcanobench run ruapehu ~/rsam_mp && aitana-bot notify ~/rsam_mp
```

`aitana-bot notify OUTDIR` picks the most recently written `results/*/alert.json` in
`OUTDIR`. It always prints a one-line summary; if the alert raised an alarm it posts the
message with the plot (`rsam_mp.png`) to the channel. `--dry-run` prints the message instead
of posting it. Every message is marked as an automated draft for duty-officer review.
