import os
import time
from pathlib import Path

import typer
from slack_sdk import WebClient

from .notify import finished_alert, format_message, latest_alert, post_to_slack

app = typer.Typer()

POLL_SECONDS = 60


@app.callback()
def main():
    """Aitana duty-officer companion."""


def _echo_summary(alert, path):
    typer.echo(
        f"{path}: peak index {alert['peak_index']} at {alert['peak_time']} UTC, "
        f"alarm={alert['alarm']}"
    )


def _slack_credentials():
    """
    Read the Slack bot token and channel ID from the environment.

    Returns
    -------
    tuple of (str, str)
        ``SLACK_BOT_TOKEN`` and ``SLACK_CHANNEL_ID``.

    Raises
    ------
    typer.Exit
        If either is unset.
    """
    token = os.environ.get("SLACK_BOT_TOKEN")
    channel = os.environ.get("SLACK_CHANNEL_ID")
    if not token or not channel:
        typer.echo(
            "Set SLACK_BOT_TOKEN and SLACK_CHANNEL_ID to post to Slack.", err=True
        )
        raise typer.Exit(1)
    return token, channel


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
    _echo_summary(alert, path)
    if not alert["alarm"] and not testbot:
        return

    text = format_message(alert)
    image = path.parent / alert["plot"]
    if dry_run:
        typer.echo(f"{text}\nplot: {image}")
        return

    token, channel = _slack_credentials()
    post_to_slack(WebClient(token=token), channel, text, image)
    typer.echo(f"Posted to {channel}.")


@app.command()
def watch(outdir: Path):
    """Keep checking OUTDIR and post each newly raised alarm to Slack.

    A finished run (alert and plot written) is posted when it raises the alarm
    and the finished run before it did not, so an alarm that stays raised over
    consecutive runs is posted once. Runs finished before the watcher started
    are not posted.
    """
    token, channel = _slack_credentials()
    client = WebClient(token=token)
    previous = finished_alert(outdir)
    while True:
        time.sleep(POLL_SECONDS)
        current = finished_alert(outdir)
        if current is None or (previous is not None and current[1] == previous[1]):
            continue
        alert, path = current
        _echo_summary(alert, path)
        if alert["alarm"] and not (previous is not None and previous[0]["alarm"]):
            image = path.parent / alert["plot"]
            post_to_slack(client, channel, format_message(alert), image)
            typer.echo(f"Posted to {channel}.")
        previous = current
