"""
Offline checks of the ruapehu_snet workflow: its job graph (Snakemake dry runs)
and, with --runslow, a real run of its store rule on synthetic coefficients.
"""

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aitana.pipelines.cli import SnakemakeBackend
from aitana.pipelines.workflows.ruapehu_snet import workflow
from aitana.storage import Storage


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


def write_snet(outdir, days, empty=()):
    """
    Write synthetic coefficients where the snet_day rule caches them.

    Parameters
    ----------
    outdir : pathlib.Path
        Workflow output directory.
    days : pd.DatetimeIndex
        UTC days to write, 144 windows each, shaped like ``SNet.compute`` output.
    empty : sequence of pd.Timestamp
        Days written without coefficients, as snet_day does without three components.

    Returns
    -------
    dict of str to xr.Dataset
        The written datasets by day.
    """
    (outdir / "snet").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    written = {}
    for day in days:
        coords = {
            "datetime": pd.date_range(day, periods=144, freq="10min"),
            "channel": np.arange(3),
            "frequency": np.geomspace(10.5, 0.195, 24),
        }
        order1 = xr.DataArray(
            rng.lognormal(size=(144, 3, 24)), dims=list(coords), coords=coords
        )
        coords["order2 frequency"] = np.geomspace(6.25, 0.195, 6)
        order2 = xr.DataArray(
            rng.lognormal(size=(144, 3, 24, 6)), dims=list(coords), coords=coords
        )
        snet = xr.Dataset() if day in empty else xr.Dataset({"order1": order1, "order2": order2})
        snet.to_netcdf(outdir / "snet" / f"{day:%Y-%m-%d}.nc")
        written[f"{day:%Y-%m-%d}"] = snet
    return written


@pytest.mark.parametrize(
    "end, start, days, partial",
    [
        # the window starting at 23:50 has not ended yet: the whole day is partial
        ("2022-03-14T00:00", None, 0, True),
        # it has, and no window of the new day has
        ("2022-03-14T00:10", None, 1, False),
        ("2022-03-14T05:30", None, 1, True),
        # backfill: every day since start
        ("2022-03-14T05:30", "2022-03-01", 13, True),
    ],
)
def test_job_graph(tmp_path, capfd, end, start, days, partial):
    config = [f"end={end}"] + ([f"start={start}"] if start else [])
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1, extra=["-n", "--config", *config]
    )
    jobs = planned_jobs(capfd.readouterr().out)
    assert jobs.get("snet_day", 0) == days
    assert jobs.get("snet_partial", 0) == int(partial)
    assert jobs["store"] == 1


@pytest.mark.slow
def test_store(tmp_path):
    """Stored coefficients match the day files, NaN for a day without data."""
    days = pd.date_range("2022-03-11", "2022-03-13")
    written = write_snet(tmp_path, days, empty=days[1:2])
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1, extra=["--config", "end=2022-03-14T00:10", "start=2022-03-11"]
    )
    assert (tmp_path / "results" / "20220314T0010" / "order1.png").exists()

    storage = Storage(
        "ruapehu", tmp_path / "storage", starttime=days[0], endtime=pd.Timestamp("2022-03-14")
    )
    store = storage.get_substore("FWVZ", "10", "HH")
    for name in ("order1", "order2"):
        stored = store(f"snet_{name}")
        stored["datetime"] = stored.indexes["datetime"].round("1min")
        stored = stored.transpose(*written["2022-03-11"][name].dims)
        assert stored.sizes["datetime"] == 3 * 144
        assert stored.sel(datetime="2022-03-12").isnull().all()
        for day in ("2022-03-11", "2022-03-13"):
            np.testing.assert_allclose(
                stored.sel(datetime=day).values, written[day][name].values
            )
