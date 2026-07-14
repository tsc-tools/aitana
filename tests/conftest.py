import inspect
import os
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

import aiobotocore.endpoint
import moto.core.botocore_stubber
import numpy as np
import pandas as pd
import pytest
import requests
import xarray as xr
from aitana.access.seismic import SDSWaveforms
from aitana.access.seismic import SeismicData as DataSource
from aitana.storage import Storage, generate_test_data, get_labels
from fastapi.testclient import TestClient
from obspy import Trace, UTCDateTime

# moto's mocked responses are built as synchronous botocore.awsrequest.AWSResponse
# objects, but s3fs talks to S3 through aiobotocore's async client, which awaits
# `.content` and reads `.raw.content.read(...)`. Neither exists on moto's stub,
# so any test using @mock_aws with s3fs/fsspec (e.g. S3Waveforms) fails with
# "TypeError: object bytes can't be used in 'await' expression" -- silently, if
# the caller swallows the exception. Patch both call sites so moto's stub
# response behaves like an async one; real (non-mocked) async responses already
# expose an awaitable `.content` and pass through unchanged.


class _AsyncRawContent:
    def __init__(self, raw):
        self._raw = raw

    async def read(self, amt=-1):
        return self._raw.read(None if amt is None or amt < 0 else amt)


moto.core.botocore_stubber.MockRawResponse.content = property(
    lambda self: _AsyncRawContent(self)
)

_orig_convert_to_response_dict = aiobotocore.endpoint.convert_to_response_dict


class _AsyncResponseWrapper:
    def __init__(self, response):
        self._response = response

    def __getattr__(self, item):
        return getattr(self._response, item)

    async def _content_prop(self):
        content = self._response.content
        return await content if inspect.isawaitable(content) else content

    @property
    def content(self):
        return self._content_prop()


async def _patched_convert_to_response_dict(http_response, operation_model):
    return await _orig_convert_to_response_dict(
        _AsyncResponseWrapper(http_response), operation_model
    )


aiobotocore.endpoint.convert_to_response_dict = _patched_convert_to_response_dict


def pytest_addoption(parser):
    parser.addoption(
        "--runwebservice",
        action="store_true",
        default=False,
        help="run webservice tests",
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line(
        "markers", "webservice: mark tests that request data from webservices"
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runwebservice"):
        skip_webservice = pytest.mark.skip(reason="need --runwebservice option to run")
        for item in items:
            if "webservice" in item.keywords:
                item.add_marker(skip_webservice)
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


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


def download_and_extract_tar(url, extract_path):
    # Download the tar file
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open("file.tar", "wb") as f:
            f.write(response.raw.read())

        # Open the tar file
        tar = tarfile.open("file.tar")

        # Extract it
        tar.extractall(path=extract_path)
        tar.close()
    else:
        print(f"Failed to download {url}")


@pytest.fixture(scope="module")
def setup_sds():
    test_data_dir = os.path.join(os.environ["HOME"], "zizou_test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(test_data_dir, "sds_test")):
        url = "https://zenodo.org/records/14942280/files/zizou_test_data.tar.gz"
        download_and_extract_tar(url, test_data_dir)
    sds_dir = os.path.join(test_data_dir, "sds_test")
    fdsn_urls = ("https://service.geonet.org.nz", "https://service-nrt.geonet.org.nz")
    sdsc = SDSWaveforms(
        sds_dir=sds_dir, fdsn_urls=fdsn_urls, staxml_dir=sds_dir, fill_value=np.nan
    )
    ds = DataSource(clients=[sdsc])
    return sds_dir, ds


@pytest.fixture(scope="module")
def setup_temporary_sds(tmp_path_factory):
    sds_dir = tmp_path_factory.mktemp("aws_test_tmp", numbered=True)
    shutil.copy(Path(__file__).parent / "data" / "MAVZ.xml", sds_dir)
    return sds_dir


@pytest.fixture
def setup_api(tmp_path_factory, monkeypatch):
    from aitana.access.api import AitanaAPI

    savedir = tmp_path_factory.mktemp("zizou_api_test_tmp", numbered=True)
    api = AitanaAPI(rootdir=savedir)

    def mockreturn(net, sta, loc, chan, start, end):
        T_x, N = 1 / 100, 60000  # 20 Hz sampling rate for 50 s signal
        t_x = np.arange(N) * T_x  # time indexes for signal
        f_i = 1 * np.arctan((t_x - t_x[N // 2]) / 2) + 5  # varying frequency
        dummy_data = np.sin(2 * np.pi * np.cumsum(f_i) * T_x)
        dummy_trace = Trace(data=dummy_data)
        dummy_trace.stats.sampling_rate = 1.0 / T_x
        dummy_trace.stats.starttime = UTCDateTime("2020-01-01T00:00:00")
        yield dummy_trace

    monkeypatch.setattr(api.datasource, "get_waveforms", mockreturn)
    client = TestClient(api.app)
    return client


@pytest.fixture(scope="package")
def storage_setup(tmp_path_factory):
    features1D = [
        "rsam",
        "dsar",
        "central_freq",
        "predom_freq",
        "bandwidth",
        "rsam_energy_prop",
    ]
    features2D = [
        ("sonogram", "sonofrequency"),
        ("ssam", "frequency"),
        ("filterbank", "fbfrequency"),
    ]

    savedir = tmp_path_factory.mktemp("vumt_test_tmp", numbered=True)
    g = Storage("volcanoes", rootdir=savedir, archive_starttime=tstart)
    c1 = g.get_substore("WIZ", "00", "HHZ")
    c2 = g.get_substore("MDR", "00", "BHZ")
    c3 = g.get_substore("MAVZ", "10", "EHZ")
    c4 = g.get_substore("MMS", "66", "BHZ")
    # Generate some fake data
    for _f in features1D:
        feat = generate_test_data(tstart=tstart, feature_names=[_f], ndays=ndays)
        for _c in g.stores:
            _c.save(feat)
            if _f == "dsar":
                _c.save_labels(get_labels(feat.dsar, float(feat.dsar.quantile(0.85))))
    for _n, _f in features2D:
        feat = generate_test_data(
            tstart=tstart,
            feature_names=[_n],
            ndays=ndays,
            nfreqs=8,
            freq_names=[_f],
            dim=2,
        )
        for _c in g.stores:
            _c.save(feat)

    alg = generate_test_data(
        tstart=tstart,
        feature_names=["autoencoder"],
        ndays=ndays,
        nfreqs=5,
        freq_names=["cluster"],
        dim=2,
    )
    c2.save(alg)
    return savedir, g


@pytest.fixture(scope="module")
def setup_storage_api(storage_setup):
    savedir, g = storage_setup
    from aitana.storage.api import TonikAPI

    ta = TonikAPI(str(savedir))
    client = TestClient(ta.app)
    g.starttime = datetime(2023, 1, 1)
    g.endtime = datetime(2023, 1, 6)
    return client, g.get_substore("MDR", "00", "BHZ")


@pytest.fixture(scope="module")
def setup_api_zarr(tmp_path_factory):
    savedir = tmp_path_factory.mktemp("vumt_test_tmp_zarr", numbered=True)
    g = Storage("volcanoes", rootdir=savedir, backend="zarr", archive_starttime=tstart)
    from aitana.storage.api import TonikAPI

    ta = TonikAPI(str(savedir), backend="zarr")
    client = TestClient(ta.app)
    g.starttime = datetime(2023, 1, 1)
    g.endtime = datetime(2023, 1, 6)
    feat = generate_test_data(
        tstart=tstart,
        feature_names=["ssam"],
        ndays=ndays,
        nfreqs=8,
        freq_names=["frequency"],
        dim=2,
    )
    c = g.get_substore("MDR", "00", "BHZ")
    c.save(feat)
    return client, c


@pytest.fixture(scope="module")
def setup_multi_dimensional(tmp_path_factory):
    tempdir = tmp_path_factory.mktemp("test_xarray2zarr_high_dimensionality")
    test_data = xr.DataArray(
        np.random.rand(3, 24, 6, 144),
        dims=["channel", "order_1", "order_2", "datetime"],
        coords={
            "datetime": pd.date_range(start="2022-07-18", periods=144, freq="10min"),
            "channel": np.arange(3),
            "order_1": np.arange(24),
            "order_2": np.arange(6),
        },
    )
    test_data_2 = xr.DataArray(
        np.random.rand(3, 24, 6, 10),
        dims=["channel", "order_1", "order_2", "datetime"],
        coords={
            "datetime": pd.date_range(start="2022-07-20", periods=10, freq="10min"),
            "channel": np.arange(3),
            "order_1": np.arange(24),
            "order_2": np.arange(6),
        },
    )
    xds = xr.Dataset({"order2": test_data})
    xds2 = xr.Dataset({"order2": test_data_2})
    return tempdir, xds, xds2
