import os
from pathlib import Path

import typer
from slack_sdk import WebClient

from .notify import format_message, latest_alert, post_to_slack

app = typer.Typer()


@app.callback()
def main():
    """Aitana duty-officer companion."""


@app.command()
def notify(
    outdir: Path,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the message instead of posting it."
    ),
    testbot: bool = typer.Option(
        False,
        "--test-bot",
        help="Test the bot by posting a message to Slack even if alert is not raised.",
    ),
):
    """Post the latest workflow alert in OUTDIR to Slack if it raised an alarm."""
    alert, path = latest_alert(outdir)
    typer.echo(
        f"{path}: peak index {alert['peak_index']} at {alert['peak_time']} UTC, "
        f"alarm={alert['alarm']}"
    )
    if not alert["alarm"] and not testbot:
        return

    text = format_message(alert)
    image = path.parent / alert["plot"]
    if dry_run:
        typer.echo(f"{text}\nplot: {image}")
        return

    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        typer.echo(
            "Set SLACK_BOT_TOKEN and SLACK_CHANNEL_ID to post to Slack.", err=True
        )
        raise typer.Exit(1)
    post_to_slack(WebClient(token=token), channel, text, image)
    typer.echo(f"Posted to {channel}.")
