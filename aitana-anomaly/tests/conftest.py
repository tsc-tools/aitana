from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from aitana.storage import Storage, generate_test_data

tstart = datetime(2023, 1, 1, 0, 0, 0)
ndays = 10


@pytest.fixture(scope="package")
def setup(tmp_path_factory):
    features1D = [
        "rsam",
        "dsar",
        "central_freq",
        "predom_freq",
        "bandwidth",
        "rsam_energy_prop",
    ]
    features2D = ["sonogram", "ssam", "filterbank"]
    features2D_freqs = ["sonofrequency", "frequency", "fbfrequency"]

    savedir = tmp_path_factory.mktemp("zizou_test_tmp", numbered=True)
    sgw = Storage("volcanoes", savedir)
    s_wiz = sgw.get_substore("WIZ", "00", "HHZ")
    s_wiz.station = "NZ.WIZ.00.HHZ"
    s_mdr = sgw.get_substore("MDR", "00", "BHZ")
    s_mdr.station = "NZ.MDR.00.BHZ"
    s_mavz = sgw.get_substore("MAVZ", "00", "EHZ")
    s_mavz.station = "NZ.MAVZ.00.EHZ"
    s_mms = sgw.get_substore("MMS", "00", "BHZ")
    s_mms.station = "NZ.MMS.00.BHZ"
    # Generate some fake data
    feat = generate_test_data(tstart=tstart, feature_names=features1D, ndays=ndays)
    for _s in [s_wiz, s_mdr, s_mavz, s_mms]:
        feat["datetime"] = feat.datetime.dt.round("10min")
        _s.save(feat)
    feat = generate_test_data(
        tstart=tstart,
        feature_names=features2D,
        freq_names=features2D_freqs,
        ndays=ndays,
        nfreqs=8,
        dim=2,
    )
    for _s in [s_wiz, s_mdr, s_mavz, s_mms]:
        feat["datetime"] = feat.datetime.dt.round("10min")
        _s.save(feat)

    alg = generate_test_data(
        tstart=tstart,
        feature_names=["autoencoder"],
        ndays=ndays,
        nfreqs=5,
        freq_names=["cluster"],
        dim=2,
    )
    s_mdr.save(alg)
    return savedir


@pytest.fixture(scope="module")
def setup_dataset(tmp_path_factory):
    npts = 3
    npts2 = 4
    vals = np.array(
        [
            [1.0, np.nan, np.nan, 3.0],
            [4.0, 5.0, np.nan, 7.0],
            [np.nan, 9.0, 10.0, 11.0],
        ]
    )
    xds_2D = xr.Dataset(
        {"ssam": (["frequency", "datetime"], vals)},
        coords={
            "frequency": np.arange(npts),
            "datetime": pd.date_range(start=tstart, periods=npts2, freq="10min"),
        },
    )

    xds_1D = xr.Dataset(
        {"rsam": (["datetime"], vals[1, :])},
        coords={
            "datetime": pd.date_range(start=tstart, periods=npts2, freq="10min"),
        },
    )
    xds_1D.attrs["resolution"] = 1 / 6.0
    xds_2D.attrs["resolution"] = 1 / 6.0

    savedir = tmp_path_factory.mktemp("zizou_dataset_test_tmp", numbered=True)
    sgw = Storage(
        "volcanoes",
        savedir,
        starttime=tstart,
        endtime=tstart + timedelta(minutes=10 * npts2),
        archive_starttime=tstart,
    )
    s_wiz = sgw.get_substore("WIZ", "00", "HHZ")
    # Generate some fake data
    s_wiz.save(xds_1D)
    s_wiz.save(xds_2D)
    return s_wiz


@pytest.fixture(scope="module")
def setup_ac(setup):
    savedir = setup
    sg = Storage(
        "volcanoes",
        rootdir=savedir,
        starttime=tstart,
        endtime=tstart + timedelta(days=ndays),
    )
    s_wiz = sg.get_substore("WIZ", "00", "HHZ")
    s_wiz.station = "NZ.WIZ.00.HHZ"
    s_wsrz = sg.get_substore("WSRZ", "00", "HHZ")
    s_wsrz.station = "NZ.WSRZ.00.HHZ"
    config = """
    autoencoder:
      layers: [2000,500,200,6]
      epochs: 5
      patience: 10
    """
    return savedir, s_wiz, s_wsrz, config
