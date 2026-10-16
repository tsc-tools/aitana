import json
import os

import pytest
from typer.testing import CliRunner

from aitana.bot import cli
from aitana.bot.notify import DISCLAIMER, format_message, latest_alert, post_to_slack


class FakeClient:
    """Records Slack uploads instead of sending them."""

    instances = []

    def __init__(self, token=None):
        self.token = token
        self.uploads = []
        FakeClient.instances.append(self)

    def files_upload_v2(self, **kwargs):
        self.uploads.append(kwargs)
        return {"ok": True}


def write_alert(outdir, tag, alarm, mtime):
    """
    Write a minimal workflow result (alert.json and a dummy plot).

    Parameters
    ----------
    outdir : Path
        Workflow output directory.
    tag : str
        Result folder name.
    alarm : bool
        Value of the alert's ``alarm`` field.
    mtime : float
        Modification time to set on alert.json.

    Returns
    -------
    Path
        Path to the alert.json.
    """
    resdir = outdir / "results" / tag
    resdir.mkdir(parents=True)
    alert = {
        "volcano": "Ruapehu",
        "seed_id": "NZ.FWVZ.10.HHZ",
        "end": "2022-03-20T00:00:00",
        "score_start": "2022-03-19T00:00:00",
        "history_start": "2022-02-18T18:00:00",
        "window_hours": 6.0,
        "threshold": 0.999,
        "peak_index": 1.0 if alarm else 0.8,
        "peak_time": "2022-03-19T17:30:00",
        "alarm": alarm,
        "plot": "rsam_mp.png",
    }
    path = resdir / "alert.json"
    path.write_text(json.dumps(alert))
    (resdir / "rsam_mp.png").write_bytes(b"png")
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture
def slack(monkeypatch):
    FakeClient.instances = []
    monkeypatch.setattr(cli, "WebClient", FakeClient)
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setenv("SLACK_CHANNEL_ID", "C123")
    return FakeClient


def test_latest_alert_is_most_recently_written(tmp_path):
    write_alert(tmp_path, "20260101T0000", alarm=False, mtime=2_000)
    newest = write_alert(tmp_path, "20220320T0000", alarm=True, mtime=3_000)
    alert, path = latest_alert(tmp_path)
    assert path == newest
    assert alert["alarm"]


def test_message():
    alert = {
        "volcano": "Ruapehu",
        "seed_id": "NZ.FWVZ.10.HHZ",
        "history_start": "2022-02-18T18:00:00",
        "window_hours": 6.0,
        "threshold": 0.999,
        "peak_index": 0.99975,
        "peak_time": "2022-03-19T17:30:00",
    }
    text = format_message(alert)
    assert "*0.9998*" in text
    assert "99.98%" in text
    assert "2022-03-19 17:30 UTC" in text
    assert "threshold 0.999" in text
    assert DISCLAIMER in text


def test_post_to_slack(tmp_path):
    client = FakeClient()
    post_to_slack(client, "C123", "hello", tmp_path / "rsam_mp.png")
    assert client.uploads == [
        {
            "channel": "C123",
            "file": str(tmp_path / "rsam_mp.png"),
            "title": "rsam_mp.png",
            "initial_comment": "hello",
        }
    ]


def test_notify_posts_on_alarm(tmp_path, slack):
    path = write_alert(tmp_path, "20220320T0000", alarm=True, mtime=1_000)
    result = CliRunner().invoke(cli.app, ["notify", str(tmp_path)])
    assert result.exit_code == 0, result.output
    (client,) = slack.instances
    (upload,) = client.uploads
    assert upload["channel"] == "C123"
    assert upload["file"] == str(path.parent / "rsam_mp.png")
    assert DISCLAIMER in upload["initial_comment"]


def test_notify_silent_without_alarm(tmp_path, slack):
    write_alert(tmp_path, "20220220T0000", alarm=False, mtime=1_000)
    result = CliRunner().invoke(cli.app, ["notify", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert slack.instances == []


def test_notify_dry_run(tmp_path, slack):
    write_alert(tmp_path, "20220320T0000", alarm=True, mtime=1_000)
    result = CliRunner().invoke(cli.app, ["notify", str(tmp_path), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert DISCLAIMER in result.output
    assert slack.instances == []


def test_notify_needs_credentials(tmp_path, slack, monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN")
    write_alert(tmp_path, "20220320T0000", alarm=True, mtime=1_000)
    result = CliRunner().invoke(cli.app, ["notify", str(tmp_path)])
    assert result.exit_code == 1
    assert slack.instances == []


class StopWatching(Exception):
    pass


def test_watch_posts_each_raised_alarm_once(tmp_path, slack, monkeypatch):
    # Already raised before the watcher starts: not posted.
    write_alert(tmp_path, "20220319T0000", alarm=True, mtime=1_000)
    unfinished = tmp_path / "results" / "20220319T0100" / "rsam_mp.png"
    runs = iter(
        [
            lambda: write_alert(tmp_path, "20220319T0010", alarm=True, mtime=2_000),
            lambda: write_alert(tmp_path, "20220319T0020", alarm=False, mtime=3_000),
            lambda: write_alert(tmp_path, "20220319T0030", alarm=True, mtime=4_000),
            lambda: write_alert(tmp_path, "20220319T0040", alarm=True, mtime=5_000),
            lambda: write_alert(tmp_path, "20220319T0050", alarm=False, mtime=6_000),
            # Raised again, but the run has not written its plot yet...
            lambda: write_alert(tmp_path, "20220319T0100", alarm=True, mtime=7_000)
            .with_name("rsam_mp.png")
            .unlink(),
            # ...until the next check.
            lambda: unfinished.write_bytes(b"png"),
        ]
    )

    def sleep(seconds):
        """Let the next workflow run finish instead of waiting."""
        try:
            next(runs)()
        except StopIteration:
            raise StopWatching

    posted = []
    monkeypatch.setattr(cli.time, "sleep", sleep)
    monkeypatch.setattr(
        cli,
        "post_to_slack",
        lambda client, channel, text, image: posted.append((image, image.exists())),
    )
    result = CliRunner().invoke(cli.app, ["watch", str(tmp_path)])

    assert isinstance(result.exception, StopWatching), result.output
    assert posted == [
        (tmp_path / "results" / tag / "rsam_mp.png", True)
        for tag in ["20220319T0030", "20220319T0100"]
    ]


def test_watch_needs_credentials(tmp_path, slack, monkeypatch):
    monkeypatch.delenv("SLACK_CHANNEL_ID")
    result = CliRunner().invoke(cli.app, ["watch", str(tmp_path)])
    assert result.exit_code == 1
    assert slack.instances == []
