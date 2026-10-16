"""
Format and post anomaly alerts written by Aitana workflows.
"""

import json
from datetime import datetime
from pathlib import Path

DISCLAIMER = "_Automated draft for duty-officer review, not an assessment._"


def latest_alert(outdir):
    """
    Load the most recently written alert summary of a workflow output directory.

    Parameters
    ----------
    outdir : str or Path
        Output directory of ``aitana-pipelines run``.

    Returns
    -------
    alert : dict
        Contents of the newest ``results/<end>/alert.json``.
    path : Path
        Path to that file.
    """
    paths = list(Path(outdir).glob("results/*/alert.json"))
    if not paths:
        raise FileNotFoundError(f"No results/*/alert.json under {outdir}")
    path = max(paths, key=lambda p: p.stat().st_mtime)
    return json.loads(path.read_text()), path


def finished_alert(outdir):
    """
    Load the newest alert summary of a workflow output directory once its run is done.

    Parameters
    ----------
    outdir : str or Path
        Output directory of ``aitana-pipelines run``.

    Returns
    -------
    tuple of (dict, Path) or None
        Like `latest_alert`, or None while there is no alert yet or the newest
        run has not written its plot.
    """
    try:
        alert, path = latest_alert(outdir)
    except FileNotFoundError:
        return None
    if not (path.parent / alert["plot"]).exists():
        return None
    return alert, path


def format_message(alert):
    """
    Turn an alert summary into a Slack mrkdwn message.

    Parameters
    ----------
    alert : dict
        Alert summary written by the workflow (all times UTC).

    Returns
    -------
    str
        Message text.
    """
    peak_time = datetime.fromisoformat(alert["peak_time"])
    history_start = datetime.fromisoformat(alert["history_start"])
    window = f"{alert['window_hours']:g} h"
    return "\n".join(
        [
            f":warning: *{alert['volcano']}: unusual RSAM pattern* ({alert['seed_id']})",
            f"Anomaly index *{alert['peak_index']:.4f}* at {peak_time:%Y-%m-%d %H:%M} UTC "
            f"(alarm threshold {alert['threshold']:g}).",
            f"The {window} of RSAM ending then match their closest look-alike since "
            f"{history_start:%Y-%m-%d} worse than {alert['peak_index']:.2%} of {window} "
            "windows in that period do (z-normalized matrix profile, so this is about "
            "shape, not amplitude).",
            DISCLAIMER,
        ]
    )


def post_to_slack(client, channel, text, image_path):
    """
    Post a message with its plot to a Slack channel.

    Parameters
    ----------
    client : slack_sdk.WebClient
        Client authenticated with a bot token (scopes ``chat:write``,
        ``files:write``).
    channel : str
        Channel ID the bot has been invited to.
    text : str
        Message text, posted as the file's comment.
    image_path : str or Path
        Plot to attach.

    Returns
    -------
    slack_sdk.web.SlackResponse
        Response of the upload.
    """
    return client.files_upload_v2(
        channel=channel,
        file=str(image_path),
        title=Path(image_path).name,
        initial_comment=text,
    )
