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
    # two reference days for the embedding and the novelty index, before the
    # computed period
    config = [f"end={end}", "fit_start=2022-01-01", "fit_end=2022-01-03",
              "novelty_start=2022-01-01", "novelty_end=2022-01-03"]
    config += [f"start={start}"] if start else []
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1, extra=["-n", "--config", *config]
    )
    jobs = planned_jobs(capfd.readouterr().out)
    assert jobs.get("snet_day", 0) == days + 2
    assert jobs.get("snet_partial", 0) == int(partial)
    assert jobs["store"] == 1
    assert jobs["fit_embedding"] == 1
    assert jobs["embed"] == 1
    assert jobs["fit_novelty"] == 1
    assert jobs["novelty"] == 1


@pytest.mark.parametrize(
    "outputs, days",
    [
        # complete: only the last lookback_hours
        (("stored", "embedding.csv", "novelty.csv"), 1),
        # interrupted before its last rule finished: still every day since start
        (("stored", "embedding.csv"), 13),
    ],
)
def test_job_graph_with_archive(tmp_path, capfd, outputs, days):
    archive = tmp_path / "results" / "archive_20220301"
    archive.mkdir(parents=True)
    for name in outputs:
        (archive / name).touch()
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1,
        extra=["-n", "--config", "end=2022-03-14T05:30", "start=2022-03-01",
               "fit_start=2022-01-01", "fit_end=2022-01-03",
               "novelty_start=2022-01-01", "novelty_end=2022-01-03"],
    )
    assert planned_jobs(capfd.readouterr().out)["snet_day"] == days + 2


@pytest.mark.slow
def test_store(tmp_path):
    """
    Stored coefficients match the day files and the stored embedding and
    novelty series match embedding.csv and novelty.csv, NaN for a day without
    data.
    """
    days = pd.date_range("2022-03-11", "2022-03-13")
    written = write_snet(tmp_path, days, empty=days[1:2])
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1,
        extra=["--config", "end=2022-03-14T00:10", "start=2022-03-11",
               "fit_start=2022-03-11", "fit_end=2022-03-14",
               "novelty_start=2022-03-11", "novelty_end=2022-03-14", "threshold=0.9"],
    )
    results = tmp_path / "results" / "archive_20220311"
    assert not list(results.glob("*.png"))

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

    stored = store("snet_umap")
    # the layout aitana-api expects of a 2-D feature
    assert stored.dims == ("component", "datetime")
    stored = stored.transpose("datetime", "component")
    stored["datetime"] = stored.indexes["datetime"].round("1min")
    csv = pd.read_csv(results / "embedding.csv", index_col=0, parse_dates=True)
    # labelled 10 min after each window's label
    assert csv.index[0] == pd.Timestamp("2022-03-11 00:10")
    stored = stored.sel(datetime=csv.index)
    assert stored.sizes["datetime"] == 3 * 144
    assert stored.sel(datetime=slice("2022-03-12 00:10", "2022-03-13 00:00")).isnull().all()
    # the other two days are complete
    assert int(stored.notnull().all("component").sum()) == 2 * 144
    np.testing.assert_allclose(stored.values, csv[["umap0", "umap1"]].values)

    csv = pd.read_csv(results / "novelty.csv", index_col=0, parse_dates=True)
    # labelled 10 min after each window's label
    assert csv.index[0] == pd.Timestamp("2022-03-11 00:10")
    assert csv.loc["2022-03-12 00:10":"2022-03-13 00:00"].isna().all().all()
    assert csv["anomaly_index"].notna().sum() == 2 * 144
    scored = csv["anomaly_index"].notna()
    assert (csv["alarm"][scored] == (csv["anomaly_index"][scored] >= 0.9)).all()
    for name, column in [("index", "anomaly_index"), ("distance", "distance"),
                         ("alarm", "alarm")]:
        stored = store(f"snet_novelty_{name}").to_series()
        stored.index = stored.index.round("1min")
        np.testing.assert_allclose(stored.reindex(csv.index).values, csv[column].values)
    threshold = store("snet_novelty_threshold").to_series()
    assert (threshold.dropna() == 0.9).all() and threshold.notna().sum() == len(csv)


@pytest.mark.slow
def test_update_after_archive(tmp_path):
    """
    A run after the archive stores only its own windows, keeps the stored
    history, and scores the windows it shares with the archive identically.
    """
    days = pd.date_range("2022-03-11", "2022-03-14")
    written = write_snet(tmp_path, days)
    reference = ["start=2022-03-11", "fit_start=2022-03-11", "fit_end=2022-03-14",
                 "novelty_start=2022-03-11", "novelty_end=2022-03-14"]
    backend = SnakemakeBackend(workflow.workflowdir, tmp_path)
    backend.execute(cores=1, extra=["--config", "end=2022-03-14T00:10", *reference])
    # 48 h, so that the update shares 03-13 with the archive
    backend.execute(
        cores=1, extra=["--config", "end=2022-03-15T00:10", "lookback_hours=48", *reference]
    )

    def read(run, name):
        return pd.read_csv(tmp_path / "results" / run / name, index_col=0, parse_dates=True)

    archive = read("archive_20220311", "novelty.csv")
    update = read("20220315T0010", "novelty.csv")
    assert len(archive) == 3 * 144
    assert len(read("20220315T0010", "embedding.csv")) == len(update) == 2 * 144
    shared = archive.index.intersection(update.index)
    assert len(shared) == 144
    pd.testing.assert_frame_equal(archive.loc[shared], update.loc[shared])

    storage = Storage(
        "ruapehu", tmp_path / "storage", starttime=days[0], endtime=pd.Timestamp("2022-03-15")
    )
    store = storage.get_substore("FWVZ", "10", "HH")
    stored = store("snet_order1")
    stored["datetime"] = stored.indexes["datetime"].round("1min")
    stored = stored.transpose(*written["2022-03-11"]["order1"].dims)
    assert stored.sizes["datetime"] == 4 * 144
    for day in ("2022-03-11", "2022-03-14"):
        np.testing.assert_allclose(
            stored.sel(datetime=day).values, written[day]["order1"].values
        )

    stored = store("snet_novelty_index").to_series()
    stored.index = stored.index.round("1min")
    expected = update["anomaly_index"].combine_first(archive["anomaly_index"])
    assert len(expected) == 4 * 144
    np.testing.assert_allclose(stored.reindex(expected.index).values, expected.values)

    stored = store("snet_umap").transpose("datetime", "component")
    stored["datetime"] = stored.indexes["datetime"].round("1min")
    history = read("archive_20220311", "embedding.csv").loc[:"2022-03-13 00:00"]
    np.testing.assert_allclose(
        stored.sel(datetime=history.index).values, history[["umap0", "umap1"]].values
    )
