import inspect
import shutil
from pathlib import Path

import aiobotocore.endpoint
import moto.core.botocore_stubber
import numpy as np
import pytest
from aitana.access import seismic
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


@pytest.fixture(scope="module")
def setup_temporary_sds(tmp_path_factory):
    sds_dir = tmp_path_factory.mktemp("aws_test_tmp", numbered=True)
    shutil.copy(Path(__file__).parent / "data" / "MAVZ.xml", sds_dir)
    seismic.STAXML_DIR = sds_dir
    return sds_dir


@pytest.fixture
def setup_api(tmp_path_factory, monkeypatch):
    from aitana.access.api import AitanaAPI

    savedir = tmp_path_factory.mktemp("aitana_api_test_tmp", numbered=True)
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
