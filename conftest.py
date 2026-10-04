"""Repo-wide pytest configuration.

This file is also loaded when a single package's tests run in isolation
(e.g. ``hatch run test:run-pytest`` inside ``aitana-access``), so it must not
import ``aitana.*`` or third-party packages at module level. Fixtures shared
between packages import what they need inside the fixture body.
"""

import os
import tarfile

import pytest


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


def download_and_extract_tar(url, extract_path):
    import requests

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
    import numpy as np
    from aitana.access import seismic
    from aitana.access.seismic import SDSWaveforms
    from aitana.access.seismic import SeismicData as DataSource

    test_data_dir = os.path.join(os.environ["HOME"], "zizou_test_data")
    os.makedirs(test_data_dir, exist_ok=True)
    if not os.path.exists(os.path.join(test_data_dir, "sds_test")):
        url = "https://zenodo.org/records/14942280/files/zizou_test_data.tar.gz"
        download_and_extract_tar(url, test_data_dir)
    sds_dir = os.path.join(test_data_dir, "sds_test")
    seismic.STAXML_DIR = sds_dir
    fdsn_urls = ("https://service.geonet.org.nz", "https://service-nrt.geonet.org.nz")
    sdsc = SDSWaveforms(sds_dir=sds_dir, fdsn_urls=fdsn_urls, fill_value=np.nan)
    ds = DataSource(clients=[sdsc])
    return sds_dir, ds
