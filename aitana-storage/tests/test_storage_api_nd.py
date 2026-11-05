"""
The feature endpoint on features with more dimensions than time and one other,
like scattering coefficients, and what it has to keep doing for 1D features.
"""
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from aitana.storage import Storage
from aitana.storage.api import StorageAPI
from fastapi.testclient import TestClient

T0 = datetime(2023, 1, 1)
PATH = "ruapehu,FWVZ,10,HH"
F1 = 10.511 / 1.189207 ** np.arange(24)     # descending, log-spaced, as scatseisnet
F2 = 6.25 / 2.0 ** np.arange(6)
DATES = pd.date_range(T0, periods=6 * 24 * 10, freq="10min")


@pytest.fixture(scope="module", params=["netcdf", "zarr"])
def api(request, tmp_path_factory):
    """
    A store of features of every shape on each backend.

    Returns
    -------
    tuple of (TestClient, aitana.storage.storage.Path)
        Client of the API and the store it reads.
    """
    root = str(tmp_path_factory.mktemp(request.param))
    rng = np.random.default_rng(0)
    c1 = {"datetime": DATES, "channel": np.arange(3), "frequency": F1}
    o1 = xr.DataArray(rng.random((DATES.size, 3, 24)) + .1, coords=c1, dims=list(c1))
    c2 = {**c1, "order2 frequency": F2}
    o2 = xr.DataArray(rng.random((DATES.size, 3, 24, 6)) + .1, coords=c2, dims=list(c2))
    # every value is its own frequency, so that a wrong label can't hide
    lab = xr.DataArray(np.tile(F1, (DATES.size, 1)), coords={"datetime": DATES, "frequency": F1},
                       dims=["datetime", "frequency"])
    umap = xr.DataArray(rng.random((2, DATES.size)), coords={"component": [0, 1], "datetime": DATES},
                        dims=["component", "datetime"])
    rsam = xr.DataArray(np.arange(1., DATES.size + 1), coords={"datetime": DATES},
                        dims=["datetime"])
    g = Storage("ruapehu", root, archive_starttime=T0, backend=request.param)
    c = g.get_substore("FWVZ", "10", "HH")
    for name, x in [("snet_order1", o1), ("snet_order2", o2), ("labelled", lab),
                    ("snet_umap", umap), ("rsam", rsam)]:
        ds = x.to_dataset(name=name)
        ds.attrs.update(starttime=str(DATES[0]), endtime=str(DATES[-1]), station="FWVZ")
        c.save(ds)
    g.starttime, g.endtime = T0, DATES[-1].to_pydatetime()
    return TestClient(StorageAPI(root, backend=request.param).app), c


def get(client, name, **kw):
    """
    Request a feature over the whole period.

    Parameters
    ----------
    client : TestClient
        Client of the API.
    name : str
        Feature name.
    **kw
        Further query parameters.

    Returns
    -------
    httpx.Response
        The response.
    """
    return client.get("/feature", params=dict(path=PATH, name=name, starttime=str(DATES[0]),
                                              endtime=str(DATES[-1]), **kw))


def frame(r):
    """
    Parse a successful CSV response.

    Parameters
    ----------
    r : httpx.Response
        Response of the feature endpoint.

    Returns
    -------
    pd.DataFrame
        Its rows.
    """
    assert r.status_code == 200, r.text
    return pd.read_csv(StringIO(r.text))


@pytest.mark.parametrize("name", ["snet_order1", "snet_order2"])
def test_too_many_dims_names_them(api, name):
    r = get(api[0], name)
    assert r.status_code == 422 and "channel" in r.text and "frequency" in r.text


def test_order1_heatmap(api):
    client, c = api
    df = frame(get(client, "snet_order1", sel="channel:1"))
    assert list(df.columns) == ["dates", "freqs", "feature"]
    want = c("snet_order1").sel(channel=1).transpose("frequency", "datetime")
    np.testing.assert_allclose(df.feature.values, want.values.ravel())
    np.testing.assert_allclose(df.freqs.values, want.frequency.values.repeat(want.datetime.size))


def test_order1_line(api):
    client, c = api
    df = frame(get(client, "snet_order1", sel=["channel:0", "frequency:2.5"]))
    assert list(df.columns) == ["dates", "feature"]
    want = c("snet_order1").sel(channel=0).sel(frequency=2.5, method="nearest")
    np.testing.assert_allclose(df.feature.values, want.values)


@pytest.mark.parametrize("sel,free", [(["channel:0", "order2 frequency:0.39"], "frequency"),
                                      (["channel:0", "frequency:2.5"], "order2 frequency")])
def test_order2_heatmaps(api, sel, free):
    client, c = api
    df = frame(get(client, "snet_order2", sel=sel))
    assert df.freqs.nunique() == c("snet_order2")[free].size


@pytest.mark.parametrize("sel", ["channel:7", "nope:1", "channel:x", "datetime:1", "frequency:99"])
def test_bad_selection(api, sel):
    assert get(api[0], "snet_order1", sel=sel).status_code == 422


@pytest.mark.parametrize("vres", [10, 24, 50])
def test_labels_survive_aggregation(api, vres):
    df = frame(get(api[0], "labelled", resolution="1D", verticalres=vres))
    assert df.freqs.nunique() == min(vres, 24)
    # a row of the canvas averages neighbouring rows: within one filter spacing
    assert np.abs(np.log2(df.feature / df.freqs)).max() < np.log2(1.19) * 24 / min(vres, 24)


def test_time_first_2d(api):
    client, c = api
    df = frame(get(client, "labelled"))
    assert df.freqs.nunique() == 24 and np.allclose(df.feature, df.freqs)


def test_embedding_unchanged_and_sliceable(api):
    client, c = api
    assert frame(get(client, "snet_umap")).freqs.nunique() == 2
    assert list(frame(get(client, "snet_umap", sel="component:1")).columns) == ["dates", "feature"]


def test_inventory_lists_coordinates(api):
    client, c = api
    info = {f["name"]: f for f in client.get("/inventory", params={"path": PATH}).json() if isinstance(f, dict)}
    assert set(info["snet_order2"]["dims"]) == {"channel", "frequency", "order2 frequency"}
    np.testing.assert_allclose(info["snet_order1"]["dims"]["frequency"], F1)


def test_slice_honours_log_and_normalise(api):
    client, c = api
    sel = ["channel:0", "frequency:2.5"]
    plain = frame(get(client, "snet_order1", sel=sel)).feature.values
    logged = frame(get(client, "snet_order1", sel=sel, log=True)).feature.values
    np.testing.assert_allclose(logged, 10 * np.log10(plain))
    normed = frame(get(client, "snet_order1", sel=sel, normalise=True)).feature.values
    assert normed.min() == 0 and normed.max() == 1


def test_stored_1d_ignores_log_and_normalise(api):
    client, c = api
    plain = get(client, "rsam")
    assert plain.status_code == 200
    assert list(frame(plain).columns) == ["dates", "feature"]
    assert get(client, "rsam", log=True, normalise=True).content == plain.content


def test_sel_on_stored_1d_is_rejected(api):
    assert get(api[0], "rsam", sel="channel:0").status_code == 422


def test_inventory_url_of_nd_feature_explains_itself(api):
    client, c = api
    info = {f["name"]: f for f in client.get("/inventory", params={"path": PATH}).json() if isinstance(f, dict)}
    r = client.get(info["snet_order1"]["url"])
    assert r.status_code == 422 and "sel=" in r.text
    assert "dims" not in info["rsam"]
