import threading
import time
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from aitana.storage import Storage, generate_test_data
from aitana.storage.ingest import IngestWorker


def test_group(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('test_experiment', rootdir, backend='netcdf')
    c = g.get_substore('site1', 'sensor1', 'channel1')
    assert c.path == os.path.join(
        rootdir, 'test_experiment/site1/sensor1/channel1')
    assert len(g.children) == 1
    cdir1 = os.path.join(rootdir, 'test_experiment', 'MDR1', '00', 'HHZ')
    cdir2 = os.path.join(rootdir, 'test_experiment', 'MDR2', '10', 'BHZ')
    for _d in [cdir1, cdir2]:
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, 'feature.nc'), 'w') as f:
            f.write('test')
    g.from_directory()
    assert len(g.children) == 3
    c = g.get_substore('MDR1', '00', 'HHZ')


def test_subgroup(tmp_path_factory):
    """
    Test storing data in different subgroups in netcdf and zarr.
    """

    startdate = datetime(2016, 1, 1)
    enddate = datetime(2016, 1, 2, 12)
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir,
                starttime=startdate, endtime=enddate)
    xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
    g.save(xdf)
    xdf.rsam.values += 100.
    g.save(xdf, group='modified')
    rsam_original = g('rsam')
    rsam_modified = g('rsam', group='modified')
    assert int(rsam_modified.mean()) == (int(rsam_original.mean()) + 100)
    g = Storage('volcanoes', rootdir=rootdir,
                starttime=startdate, endtime=enddate,
                backend='zarr')
    xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
    g.save(xdf)
    xdf.rsam.values += 100.
    g.save(xdf, group='modified')
    rsam_original = g('rsam')
    rsam_modified = g('rsam', group='modified')
    assert int(rsam_modified.mean()) == (int(rsam_original.mean()) + 100)


def test_non_existant_feature(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('test_experiment', rootdir)
    c = g.get_substore('site1', 'sensor1', 'channel1')
    g.starttime = datetime(2016, 1, 1)
    g.endtime = datetime(2016, 1, 2)
    with pytest.raises(FileNotFoundError):
        c('non_existant_feature')


def test_from_directory(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('test_experiment', rootdir, backend='netcdf')
    c = g.get_substore('site1', 'sensor1', 'channel1')
    assert c.path == os.path.join(rootdir, 'test_experiment', 'site1',
                                  'sensor1', 'channel1')
    assert len(g.children) == 1
    cdir1 = os.path.join(rootdir, 'test_experiment', 'MDR1', '00', 'HHZ')
    cdir2 = os.path.join(rootdir, 'test_experiment', 'MDR2', '10', 'BHZ')
    for _d in [cdir1, cdir2]:
        os.makedirs(_d, exist_ok=True)
        with open(os.path.join(_d, 'feature.nc'), 'w') as f:
            f.write('test')
    g.from_directory()
    assert len(g.children) == 3
    c = g.get_substore('MDR1', '00', 'HHZ')


def test_to_dict(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('test_experiment', rootdir)
    c = g.get_substore('site1', 'sensor1', 'channel1')
    c1 = g.get_substore('MDR1', '00', 'HHZ')
    c2 = g.get_substore('MDR2', '10', 'BHZ')
    assert len(g.children) == 3
    for _s in [c1, c2]:
        with open(os.path.join(_s.path, 'feature.nc'), 'w') as f:
            f.write('test')
    _j = g.to_dict()
    assert _j['test_experiment']['MDR1']['00']['HHZ'] == {'feature': 'feature'}


def test_call_multiple_days(tmp_path_factory):
    startdate = datetime(2016, 1, 1)
    enddate = datetime(2016, 1, 2, 12)

    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir).get_substore()
    xdf = generate_test_data(dim=1, ndays=20, tstart=startdate)
    g.save(xdf)
    g.starttime = startdate
    g.endtime = enddate
    rsam = g('rsam')
    xd_index = dict(datetime=slice(startdate, enddate))
    np.testing.assert_array_almost_equal(
        rsam.loc[startdate:enddate], xdf.rsam.loc[xd_index], 5)
    # Check datetime range is correct
    first_time = pd.to_datetime(rsam.datetime.values[0])
    last_time = pd.to_datetime(rsam.datetime.values[-1])
    assert pd.to_datetime(startdate) == first_time
    assert pd.to_datetime(enddate) == last_time


def test_call_single_day(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir).get_substore()
    startdate = datetime(2016, 1, 2, 1)
    enddate = datetime(2016, 1, 2, 12)
    xdf = generate_test_data(dim=1, tstart=startdate)
    g.save(xdf)
    g1 = Storage('volcanoes', rootdir=rootdir)
    g1.starttime = startdate
    g1.endtime = enddate
    rsam = g1.get_substore()('rsam')
    # Check datetime range is correct
    first_time = pd.to_datetime(rsam.datetime.values[0])
    last_time = pd.to_datetime(rsam.datetime.values[-1])
    assert pd.to_datetime(startdate) == first_time
    assert pd.to_datetime(enddate) == last_time


def test_call_single_datapoint(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2016, 1, 2, 1)
    enddate = startdate
    g.starttime = startdate
    g.endtime = enddate
    xdf = generate_test_data(dim=1, tstart=startdate)
    g.save(xdf)
    rsam = g('rsam')
    assert xdf.rsam.loc[dict(datetime='2016-01-02T01:00:00')].item() == \
        rsam.item()


def test_shape(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    tstart = datetime(2016, 1, 1)
    g = Storage('volcanoes', rootdir=rootdir, archive_starttime=tstart)
    xdf = generate_test_data(dim=2, intervals=20, tstart=tstart)
    g.save(xdf, mode='w')
    rsam_shape = g.shape('ssam')
    assert rsam_shape['datetime'] == 20
    assert rsam_shape['frequency'] == 10

    g1 = Storage('volcanoes', rootdir=rootdir,
                 backend='zarr', archive_starttime=tstart)
    g1.save(xdf, chunk_size=10)
    rsam_shape = g1.shape('ssam')
    assert rsam_shape['datetime'] == 20
    assert rsam_shape['frequency'] == 10


@pytest.mark.parametrize("backend", ["netcdf", "zarr"])
def test_feature_info_counts_records(tmp_path_factory, backend):
    """
    Pre-allocated slots and gaps are not records, and a rewrite shows at once.
    """
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir, backend=backend,
                archive_starttime=datetime(2022, 1, 1))
    dates = pd.date_range('2022-03-19T16:00', periods=6, freq='10min')
    alarm = xr.DataArray([0., 1., np.nan, 1., 0., 0.], coords={'datetime': dates},
                         dims='datetime', name='alarm')
    g.save(alarm.to_dataset())
    assert g.get_feature_info('alarm') == {
        'name': 'alarm', 'recordCount': 5,
        'earliestRecord': '2022-03-19T16:00:00',
        'latestRecord': '2022-03-19T16:50:00'}

    # fill the gap and add one sample; file times are only as fine as the
    # kernel's clock tick, so let one pass before rewriting
    time.sleep(0.02)
    later = pd.date_range('2022-03-19T16:20', periods=5, freq='10min')
    g.save(xr.DataArray([1., 1., 0., 0., 1.], coords={'datetime': later},
                        dims='datetime', name='alarm').to_dataset())
    info = g.get_feature_info('alarm')
    assert (info['recordCount'], info['latestRecord']) == (7, '2022-03-19T17:00:00')

    # a 2D timestamp is a record if any frequency holds data
    spec = xr.DataArray([[1., np.nan, np.nan], [np.nan, np.nan, 2.]],
                        coords={'frequency': [1., 2.], 'datetime': dates[:3]},
                        dims=('frequency', 'datetime'), name='spec')
    g.save(spec.to_dataset())
    info = g.get_feature_info('spec')
    assert (info['recordCount'], info['earliestRecord'], info['latestRecord']) == (
        2, '2022-03-19T16:00:00', '2022-03-19T16:20:00')


def test_attributes_only(tmp_path_factory):
    rootdir = tmp_path_factory.mktemp('data')
    tstart = datetime(2016, 1, 1)
    g = Storage('volcanoes', rootdir=rootdir, archive_starttime=tstart)
    xdf = generate_test_data(dim=1, intervals=20, tstart=tstart)
    g.save(xdf, mode='w')
    ret = g('rsam', metadata=True)
    assert ret['update_log'].values[-1] <= np.datetime64(
        datetime.now(timezone.utc))
    assert len(ret['update_log'].values) == 1


def test_storage_ingest_worker_queue(tmp_path_factory):
    base = tmp_path_factory.mktemp('ingest_queue')
    queue_dir = base / 'queue'
    archive_dir = base / 'archive'

    storage = Storage('test', rootdir=archive_dir, backend='zarr',
                      ingest_config={'queue_path': str(queue_dir)})
    substore = storage.get_substore('MDR', '00', 'HHZ')

    data = generate_test_data(dim=1, intervals=3, freq='1h',
                              tstart=datetime(2022, 7, 18, 0, 0, 0),
                              add_nans=False)

    substore.save(data)
    messages_dir = queue_dir / 'messages'
    assert messages_dir.exists() and any(messages_dir.iterdir())

    processed = storage.run_ingest_once()
    assert processed == 1
    assert not any(messages_dir.iterdir())

    rsam_path = os.path.join(substore.path, 'rsam.zarr')
    ds = xr.open_zarr(rsam_path, group='original', chunks=None)
    np.testing.assert_array_equal(ds['rsam'].loc[dict(
        datetime=data['rsam'].datetime)].values, data['rsam'].values)
    ds.close()
    assert storage.run_ingest_once() == 0


def test_ingest_worker_run_forever_stops(tmp_path):
    worker = IngestWorker(queue_path=str(tmp_path), poll_interval=0.05)

    call_count = {"value": 0}

    def fake_run_once():
        call_count["value"] += 1
        return 0  # no messages processed

    worker.run_once = fake_run_once  # monkeypatch

    stop_event = threading.Event()
    thread = threading.Thread(
        target=worker.run_forever,
        kwargs={"stop_event": stop_event},
        daemon=True,
    )
    thread.start()

    time.sleep(0.15)
    stop_event.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert call_count["value"] >= 1


def test_ingest_worker_with_data(tmp_path_factory):
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('aitana.storage.ingest')
    rootdir = tmp_path_factory.mktemp('data')
    queue_dir = tmp_path_factory.mktemp('ingest_queue')
    storage = Storage('test', rootdir=rootdir, backend='zarr',
                      ingest_config={'queue_path': str(queue_dir)},
                      starttime=datetime(2022, 7, 1, 0, 0, 0),
                      endtime=datetime(2022, 7, 31, 0, 0, 0),
                      archive_starttime=datetime(2022, 7, 1, 0, 0, 0))
    storage.start_ingest_worker(poll_interval=0.5)
    substore = storage.get_substore('MDR', '00', 'HHZ')
    data1 = generate_test_data(dim=1, intervals=3, freq='1h',
                               tstart=datetime(2022, 7, 18, 0, 0, 0),
                               add_nans=False)
    substore.save(data1)
    time.sleep(1)
    data2 = generate_test_data(dim=1, intervals=3, freq='1h',
                               tstart=datetime(2022, 7, 18, 2, 0, 0),
                               add_nans=False)
    substore.save(data2)
    time.sleep(5)
    xdf_test = substore('rsam')
    np.testing.assert_array_equal(xdf_test.loc[{'datetime': data2.datetime}].values,
                                  data2['rsam'].values)


@pytest.mark.parametrize("backend", ["netcdf", "zarr"])
def test_off_grid_times(tmp_path_factory, backend):
    """Time stamps a few milliseconds off the grid are kept and read back on it."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir, backend=backend)
    c = g.get_substore('WIZ', '00', 'HHZ')
    tstart = datetime(2023, 1, 1)
    xdf = generate_test_data(dim=1, ndays=1, tstart=tstart, feature_names=['rsam'],
                             add_nans=False)
    grid = xdf.datetime.values
    jitter = np.random.default_rng(0).integers(-5000, 5000, len(grid))
    c.save(xdf.assign_coords(datetime=grid + jitter.astype('timedelta64[us]')))

    g.starttime = tstart
    g.endtime = datetime(2023, 1, 1, 23, 50)
    rsam = c('rsam')
    np.testing.assert_array_equal(rsam.datetime.values, grid)
    np.testing.assert_array_equal(rsam.values, xdf['rsam'].values)

    # what a query returns can be saved again without losing samples
    c.save(rsam.to_dataset(name='rsam_copy'))
    np.testing.assert_array_equal(c('rsam_copy').values, xdf['rsam'].values)


def test_drifted_archive(tmp_path_factory):
    """An archive whose stored resolution is slightly off reads back on the grid."""
    from aitana.storage.storage import AitanaDataset

    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir, backend='netcdf')
    c = g.get_substore('WIZ', '00', 'HHZ')
    tstart = datetime(2023, 1, 1)
    xdf = generate_test_data(dim=1, ndays=1, tstart=tstart, feature_names=['rsam'],
                             add_nans=False)
    # A resolution inferred from float-decoded times, as stored before it was
    # rounded; over the 23 years since the archive start, the axis drifts by 0.7 s
    c.save(xdf, resolution=10 / 60 * (1 - 1e-9))

    g.starttime = tstart
    g.endtime = datetime(2023, 1, 1, 23, 50)
    np.testing.assert_array_equal(c('rsam').datetime.values, xdf.datetime.values)
    assert c.get_feature_info('rsam')['earliestRecord'] == '2023-01-01T00:00:00'
    assert AitanaDataset(c, ['rsam']).datetimes[0] == np.datetime64('2023-01-01')


@pytest.mark.parametrize("backend", ["netcdf", "zarr"])
def test_phase(tmp_path_factory, backend):
    """An archive's grid follows the data's phase; data with another phase raise."""
    rootdir = tmp_path_factory.mktemp('data')
    # a 10 s grid from the default archive start in 2000 would have 72M slots
    g = Storage('volcanoes', rootdir=rootdir, backend=backend,
                archive_starttime=datetime(2023, 1, 1))
    c = g.get_substore('WIZ', '00', 'HHZ')
    # stamped at the centres of 10 s windows, half a step off midnight
    centres = pd.date_range('2023-01-01 00:00:05', periods=12, freq='10s')
    c.save(xr.Dataset({'ssam': ('datetime', np.arange(12.0))},
                      coords={'datetime': centres}))

    g.starttime = datetime(2023, 1, 1)
    g.endtime = datetime(2023, 1, 1, 0, 3)
    ssam = c('ssam').dropna('datetime')
    np.testing.assert_array_equal(ssam.datetime.values, centres.values)
    np.testing.assert_array_equal(ssam.values, np.arange(12.0))

    starts = centres - pd.Timedelta(5, 's')
    with pytest.raises(ValueError, match="off the grid"):
        c.save(xr.Dataset({'ssam': ('datetime', np.zeros(12))},
                          coords={'datetime': starts}))
    np.testing.assert_array_equal(c('ssam').dropna('datetime').values, np.arange(12.0))
