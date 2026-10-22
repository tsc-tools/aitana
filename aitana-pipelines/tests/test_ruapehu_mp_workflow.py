"""
Offline checks of the ruapehu_mp workflow: its job graph (Snakemake dry runs)
and, with --runslow, real runs on synthetic RSAM.
"""

import re

import numpy as np
import pandas as pd
import pytest

from aitana.anomaly.evaluation import quiet_series
from aitana.pipelines.cli import SnakemakeBackend
from aitana.pipelines.workflows.ruapehu_mp import workflow


def planned_jobs(output):
    """
    Parse the job counts from a Snakemake dry-run "Job stats" table.

    Parameters
    ----------
    output : str
        Captured stdout/stderr of the dry run.

    Returns
    -------
    dict of str to int
        Number of planned jobs per rule.
    """
    return {rule: int(n) for rule, n in re.findall(r"^(\w+)\s+(\d+)$", output, re.M)}


def write_rsam(outdir, days, empty=()):
    """
    Write synthetic RSAM day files where the rsam_day rule caches them.

    Parameters
    ----------
    outdir : pathlib.Path
        Workflow output directory.
    days : pd.DatetimeIndex
        UTC days to write, one file of 144 10-min samples each.
    empty : sequence of pd.Timestamp
        Days written without samples, as rsam_day does when there is no data.
    """
    (outdir / "rsam").mkdir(parents=True, exist_ok=True)
    rsam = 10 ** (1 + 0.2 * quiet_series(144 * len(days), rng=0).to_numpy())
    for i, day in enumerate(days):
        series = pd.Series(
            rsam[144 * i : 144 * (i + 1)],
            index=pd.date_range(day, periods=144, freq="10min"),
            name="rsam",
        )
        if day in empty:
            series = series.iloc[:0]
        series.rename_axis("datetime").to_csv(outdir / "rsam" / f"{day:%Y-%m-%d}.csv")


def read_index(path):
    """Read a workflow's index CSV."""
    return pd.read_csv(path, index_col=0, parse_dates=True)


@pytest.mark.parametrize(
    "end, start, archived, partial, days",
    [
        # 28 days of history plus the 6-h window lead and the scored day
        ("2022-03-14T00:00", None, False, False, 30),
        ("2022-03-14T05:30", None, False, True, 30),
        # building the archive: history before 2022-03-01, then every day since
        ("2022-03-14T05:30", "2022-03-01", False, True, 42),
        # the archive exists, so only the last 24 h are scored
        ("2022-03-14T05:30", "2022-03-01", True, True, 30),
    ],
)
def test_job_graph(tmp_path, capfd, end, start, archived, partial, days):
    if archived:
        (tmp_path / "results").mkdir()
        (tmp_path / "results" / "archive_20220301.csv").touch()
    config = [f"end={end}"] + ([f"start={start}"] if start else [])
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1, extra=["-n", "--config", *config]
    )
    jobs = planned_jobs(capfd.readouterr().out)
    assert jobs["rsam_day"] == days
    assert jobs.get("rsam_partial", 0) == int(partial)
    assert jobs["matrix_profile"] == 1


@pytest.mark.slow
def test_archive_day_matches_plain_run(tmp_path):
    """An archived day is scored exactly as a run ending at the next midnight."""
    write_rsam(tmp_path, pd.date_range("2022-02-11", "2022-03-13"))
    backend = SnakemakeBackend(workflow.workflowdir, tmp_path)
    backend.execute(cores=1, extra=["--config", "end=2022-03-14T00:00", "start=2022-03-12"])
    backend.execute(cores=1, extra=["--config", "end=2022-03-13T00:00"])

    columns = ["anomaly_index", "matrix_profile", "alarm"]
    archive = read_index(tmp_path / "results" / "archive_20220312.csv")
    plain = read_index(tmp_path / "results" / "20220313T0000" / "index.csv")
    day = archive.loc["2022-03-12", columns]
    assert day["anomaly_index"].notna().sum() > 100
    pd.testing.assert_frame_equal(day, plain.loc["2022-03-12", columns])
    assert archive.loc["2022-03-13", "anomaly_index"].notna().sum() > 100


@pytest.mark.slow
def test_archive_skips_days_without_history(tmp_path):
    """A day without usable history is left unscored instead of failing the archive."""
    days = pd.date_range("2022-02-11", "2022-03-13")
    write_rsam(tmp_path, days, empty=days[days < "2022-03-12"])
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1, extra=["--config", "end=2022-03-14T00:00", "start=2022-03-12"]
    )
    archive = read_index(tmp_path / "results" / "archive_20220312.csv")
    assert archive.loc["2022-03-12", "anomaly_index"].isna().all()
    assert np.isfinite(archive.loc["2022-03-13", "anomaly_index"]).any()
