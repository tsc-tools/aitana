"""Checks of the ``run`` command's daemon mode, without running Snakemake."""

from typer.testing import CliRunner

from aitana.pipelines import cli


def test_run_cron(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        cli.SnakemakeBackend,
        "execute",
        lambda self, cores, extra: calls.append((cores, extra)),
    )
    triggers = []

    def start(scheduler):
        """Fire the scheduled job once instead of blocking."""
        (job,) = scheduler.get_jobs()
        triggers.append(job.trigger)
        job.func()

    monkeypatch.setattr(cli.BlockingScheduler, "start", start)
    args = ["run", "ruapehu", str(tmp_path), "--cron", "*/10 * * * *", "--cores", "2"]
    result = CliRunner().invoke(cli.app, args + ["--config", "end=now"])

    assert result.exit_code == 0, result.output
    (trigger,) = triggers
    assert str(trigger.timezone) == "UTC"
    assert str(trigger.fields[trigger.FIELD_NAMES.index("minute")]) == "*/10"
    assert calls == [(2, ["--config", "end=now"])]


def test_clean(tmp_path):
    result = CliRunner().invoke(cli.app, ["clean", "ruapehu", str(tmp_path)])
    assert result.exit_code == 0, result.output
