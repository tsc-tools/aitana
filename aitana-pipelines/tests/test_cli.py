"""Checks of the ``run`` and ``clean`` commands, without running workflows."""

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
    args = ["run", "ruapehu", str(tmp_path), "--workflow", "ruapehu_mp"]
    args += ["--cron", "*/10 * * * *", "--cores", "2"]
    result = CliRunner().invoke(cli.app, args + ["--config", "end=now"])

    assert result.exit_code == 0, result.output
    (trigger,) = triggers
    assert str(trigger.timezone) == "UTC"
    assert str(trigger.fields[trigger.FIELD_NAMES.index("minute")]) == "*/10"
    assert calls == [(2, ["--config", "end=now"])]


def test_clean(tmp_path):
    args = ["clean", "ruapehu", str(tmp_path), "--workflow", "ruapehu_mp"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, result.output


def test_run_needs_workflow(tmp_path):
    """Workflows of one volcano would overwrite each other's files in outdir."""
    result = CliRunner().invoke(cli.app, ["run", "ruapehu", str(tmp_path)])
    assert result.exit_code == 1
    assert "--workflow" in result.output
    assert not list(tmp_path.iterdir())
