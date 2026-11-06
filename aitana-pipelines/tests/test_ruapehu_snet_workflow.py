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


def write_snet(outdir, days, empty=(), no_horizontals=()):
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
    no_horizontals : sequence of pd.Timestamp
        Days whose coefficients of channels 1 and 2 are NaN.

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
        if day in no_horizontals:
            snet = snet.where(snet.channel == 0)
        snet.to_netcdf(outdir / "snet" / f"{day:%Y-%m-%d}.nc")
        written[f"{day:%Y-%m-%d}"] = snet
    return written


# The embedding and novelty reference lie within the last lookback_hours of the
# dry runs below, so that the day files planned are those the run itself covers.
REFERENCE = ["fit_start=2022-03-12", "novelty_start=2022-03-12", "novelty_end=2022-03-13"]


@pytest.mark.parametrize(
    "end, start, days, partial",
    [
        # the window starting at 23:50 has not ended yet: the whole day is partial
        ("2022-03-14T00:00", None, 2, True),
        # it has, and no window of the new day has
        ("2022-03-14T00:10", None, 3, False),
        ("2022-03-14T05:30", None, 3, True),
        # backfill: every day since start
        ("2022-03-14T05:30", "2022-03-01", 13, True),
    ],
)
def test_job_graph(tmp_path, capfd, end, start, days, partial):
    config = [f"end={end}", *REFERENCE]
    config += [f"start={start}"] if start else []
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1, extra=["-n", "--config", *config]
    )
    jobs = planned_jobs(capfd.readouterr().out)
    assert jobs.get("snet_day", 0) == days
    assert jobs.get("snet_partial", 0) == int(partial)
    assert jobs["store"] == 1
    assert jobs["embed"] == 1
    assert jobs["fit_novelty"] == 1
    assert jobs["novelty"] == 1


@pytest.mark.parametrize(
    "outputs, channels, days",
    [
        # complete: only the last lookback_hours
        (("stored", "novelty.csv"), "[0]", 3),
        # interrupted before its last rule finished: still every day since start
        (("stored",), "[0]", 13),
        # complete, but of other channels: its scores mustn't mix with theirs
        (("stored", "novelty.csv"), "[0,1,2]", 13),
    ],
)
def test_job_graph_with_archive(tmp_path, capfd, outputs, channels, days):
    archive = tmp_path / "results" / "archive_20220301_ch0"
    archive.mkdir(parents=True)
    for name in outputs:
        (archive / name).touch()
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1,
        extra=["-n", "--config", "end=2022-03-14T05:30", "start=2022-03-01",
               f"channels={channels}", *REFERENCE],
    )
    assert planned_jobs(capfd.readouterr().out)["snet_day"] == days


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
               "fit_start=2022-03-11",
               "novelty_start=2022-03-11", "novelty_end=2022-03-14", "threshold=0.9"],
    )
    results = tmp_path / "results" / "archive_20220311_ch0"
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
    # the layout aitana-storage-api expects of a 2-D feature
    assert stored.dims == ("component", "datetime")
    stored = stored.transpose("datetime", "component")
    stored["datetime"] = stored.indexes["datetime"].round("1min")
    csv = pd.read_csv(tmp_path / "results" / "embedding.csv", index_col=0, parse_dates=True)
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
@pytest.mark.parametrize("channels, scored", [("[0]", 3 * 144), ("[0,1,2]", 2 * 144)])
def test_channels(tmp_path, channels, scored):
    """
    The embedding and the novelty index use the configured channels alone, while
    the stored coefficients keep all of them: without horizontals, a day still
    counts for Z.
    """
    days = pd.date_range("2022-03-11", "2022-03-13")
    write_snet(tmp_path, days, no_horizontals=days[2:])
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1,
        extra=["--config", "end=2022-03-14T00:10", "start=2022-03-11", f"channels={channels}",
               "fit_start=2022-03-11", "novelty_start=2022-03-11", "novelty_end=2022-03-13"],
    )
    tag = channels.strip("[]").replace(",", "")
    novelty = pd.read_csv(tmp_path / "results" / f"archive_20220311_ch{tag}" / "novelty.csv",
                          index_col=0, parse_dates=True)
    assert novelty["anomaly_index"].notna().sum() == scored
    embedding = pd.read_csv(tmp_path / "results" / "embedding.csv", index_col=0, parse_dates=True)
    assert embedding["umap0"].notna().sum() == scored

    storage = Storage(
        "ruapehu", tmp_path / "storage", starttime=days[0], endtime=pd.Timestamp("2022-03-14")
    )
    stored = storage.get_substore("FWVZ", "10", "HH")("snet_order1")
    assert stored.sizes["channel"] == 3


@pytest.mark.slow
def test_update_after_archive(tmp_path):
    """
    A run after the archive stores only its own coefficients and novelty scores,
    keeps their history, and scores the windows it shares with the archive
    identically. A new day replaces the whole embedding, also in storage, and
    keeps no earlier fit; a later run without a new day leaves it alone.
    """
    days = pd.date_range("2022-03-11", "2022-03-14")
    written = write_snet(tmp_path, days[:3])
    # as deployed; the last day's file then has to be newer than the embedding
    triggers = ["--rerun-triggers", "mtime"]
    reference = ["start=2022-03-11", "fit_start=2022-03-11",
                 "novelty_start=2022-03-11", "novelty_end=2022-03-14"]
    backend = SnakemakeBackend(workflow.workflowdir, tmp_path)
    backend.execute(cores=1, extra=[*triggers, "--config", "end=2022-03-14T00:10", *reference])

    def read(run, name):
        return pd.read_csv(tmp_path / "results" / run / name, index_col=0, parse_dates=True)

    first_fit = pd.read_csv(tmp_path / "results" / "embedding.csv", index_col=0, parse_dates=True)
    assert len(first_fit) == 3 * 144

    written |= write_snet(tmp_path, days[3:])
    # 48 h, so that the update shares 03-13 with the archive
    backend.execute(
        cores=1,
        extra=[*triggers, "--config", "end=2022-03-15T00:10", "lookback_hours=48", *reference],
    )

    archive = read("archive_20220311_ch0", "novelty.csv")
    update = read("20220315T0010", "novelty.csv")
    assert len(archive) == 3 * 144
    assert len(update) == 2 * 144
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

    second_fit = pd.read_csv(tmp_path / "results" / "embedding.csv", index_col=0, parse_dates=True)
    assert len(second_fit) == 4 * 144
    columns = ["umap0", "umap1"]
    # the refit moved the windows of the first fit, so comparing with it means something
    assert not np.allclose(second_fit.loc[first_fit.index, columns], first_fit[columns])
    stored = store("snet_umap").transpose("datetime", "component")
    stored["datetime"] = stored.indexes["datetime"].round("1min")
    np.testing.assert_allclose(
        stored.sel(datetime=second_fit.index).values, second_fit[columns].values
    )
    assert not list((tmp_path / "model").glob("embedding*"))


@pytest.mark.slow
def test_no_refit_without_new_day(tmp_path, capfd):
    """Runs during a day update the novelty index but don't refit the embedding."""
    days = pd.date_range("2022-03-11", "2022-03-13")
    write_snet(tmp_path, days)
    config = ["--config", "fit_start=2022-03-11", "novelty_start=2022-03-11",
              "novelty_end=2022-03-14"]
    backend = SnakemakeBackend(workflow.workflowdir, tmp_path)
    backend.execute(cores=1, extra=["--rerun-triggers", "mtime", *config, "end=2022-03-14T00:10"])
    capfd.readouterr()
    backend.execute(
        cores=1, extra=["-n", "--rerun-triggers", "mtime", *config, "end=2022-03-14T05:30"]
    )
    jobs = planned_jobs(capfd.readouterr().out)
    assert jobs["novelty"] == 1
    assert "embed" not in jobs
