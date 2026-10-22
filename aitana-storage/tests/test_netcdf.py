import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr
import pandas as pd
import threading
import time

from aitana.storage import Storage, generate_test_data
from aitana.storage.xarray2netcdf import xarray2netcdf
from aitana.storage.xarray2zarr import xarray2zarr


def test_xarray2netcdf(tmp_path_factory):
    """
    Test writing xarray data to hdf5.
    """
    xdf = generate_test_data(
        dim=2, ndays=3, tstart=datetime(2022, 7, 18, 0, 0, 0))
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime.fromisoformat(xdf.attrs['starttime']),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='netcdf')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf)

    xdf_test = c('ssam')
    np.testing.assert_array_equal(xdf['ssam'].values,
                                  xdf_test.values)
    np.testing.assert_array_equal(xdf['frequency'].values,
                                  np.squeeze(xdf_test['frequency'].values))
    # minor differences can occur on the level of nanoseconds; ensure
    # differences are less than 1 microsecond
    dt = np.abs((xdf_test['datetime'].values - xdf['datetime'].values)).max()
    assert dt < np.timedelta64(1, 'us')


def test_xarray2netcdf_archive_starttime(tmp_path_factory):
    xdf = generate_test_data(
        dim=1, ndays=3, tstart=datetime(2022, 7, 18, 0, 0, 0))
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime(2000, 1, 1),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='netcdf', archive_starttime=datetime(2022, 1, 1))
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf)

    xdf_test = c('rsam')
    assert np.all(
        np.isnan(xdf_test.loc['2000-01-01':'2022-07-17T23:50:00'].data))
    nitems = int((datetime(2022, 7, 18, 0, 0, 0) -
                 datetime(2022, 1, 1))/timedelta(minutes=10))
    assert xdf_test.loc['2000-01-01':'2022-07-17T23:50:00'].shape[0] == nitems


def test_xarray2netcdf_merge_arrays(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 19, 12, 0, 0)
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=start, add_nans=False)
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=end, add_nans=False)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='netcdf', archive_starttime=datetime(2022, 8, 1))
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf2)
    c.save(xdf1)
    xdf_test = c('rsam')
    assert xdf_test.isnull().sum() == 24
    assert xdf_test.loc['2022-07-18T08:00:00'] == xdf1['rsam'].loc['2022-07-18T08:00:00']
    assert xdf_test.loc['2022-07-20T11:50:00'] == xdf2['rsam'].loc['2022-07-20T11:50:00']


def test_xarray2netcdf_resolution(tmp_path_factory):
    xdf = generate_test_data(dim=1, ndays=1, tstart=datetime(2022, 7, 18, 0, 0, 0),
                             add_nans=False)
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime(2000, 1, 1),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='netcdf', archive_starttime=datetime(2022, 7, 18))
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf, resolution=0.1)

    xdf_test = c('rsam')
    xdf_test_meta = c('rsam', metadata=True)
    assert xdf_test.loc['2022-07-18T00:12:00'] == xdf['rsam'].loc['2022-07-18T00:10:00']
    assert np.isnan(xdf_test.loc['2022-07-18T00:06:00'].data)
    assert xdf_test_meta['resolution'][()] == 0.1


def test_xarray2netcdf_attributes(tmp_path_factory):
    starttime = datetime(2022, 7, 18, 0, 0, 0)
    xdf = generate_test_data(dim=1, ndays=1, tstart=starttime,
                             add_nans=False)
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime(2000, 1, 1),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='netcdf', archive_starttime=starttime)
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf)
    xdf_test = c('rsam')
    assert xdf_test.attrs['station'] == xdf.attrs['station']
    assert xdf_test.attrs['feature'] == 'rsam'


def test_xarray2netcdf_metadata(tmp_path_factory):
    starttime = datetime(2022, 7, 18, 0, 0, 0)
    xdf = generate_test_data(dim=1, ndays=1, tstart=starttime,
                             add_nans=False)
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=datetime(2000, 1, 1),
                endtime=datetime.fromisoformat(xdf.attrs['endtime']),
                backend='netcdf', archive_starttime=starttime)
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf)
    starttime = datetime(2022, 7, 19, 0, 0, 0)
    xdf = generate_test_data(dim=1, ndays=1, tstart=starttime,
                             add_nans=False)
    c.save(xdf)
    xdf_test = c('rsam', metadata=True)
    now = np.datetime64(datetime.now(timezone.utc))
    assert xdf_test['update_log'].values[-1] <= now
    assert xdf_test['last_datapoint'].values[-1] == xdf.datetime.values[-1]
    assert xdf_test['resolution'][()] == 10./60.
    assert len(xdf_test['update_log'].values) == 2
    assert len(xdf_test['last_datapoint'].values) == 2


def test_xarray2netcdf_with_gaps(tmp_path_factory):
    """
    Test writing xarray data to hdf5 with gaps.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    start = datetime(2022, 7, 18, 8, 0, 0)
    end = datetime(2022, 7, 19, 12, 0, 0)
    xdf1 = generate_test_data(dim=1, ndays=1, tstart=start, add_nans=False)
    xdf2 = generate_test_data(dim=1, ndays=1, tstart=end, add_nans=False)
    g = Storage('test_experiment', rootdir=temp_dir,
                starttime=start, endtime=end + timedelta(days=1),
                backend='netcdf')
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(xdf1)
    c.save(xdf2)
    xdf_test = c('rsam')
    assert xdf_test.isnull().sum() == 24


def test_xarray2netcdf_zeros(tmp_path_factory):
    """
    Zeros are data, not missing values: a 0/1 alarm series reads back as
    written, and only the slots that were never written are NaN.
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    dates = pd.date_range('2022-03-19T16:00', periods=6, freq='10min')
    alarm = xr.DataArray([0., 1., 0., np.nan, 1., 0.], coords={'datetime': dates},
                         dims='datetime', name='alarm')
    g = Storage('test_experiment', rootdir=temp_dir, starttime=dates[0],
                endtime=dates[-1], backend='netcdf',
                archive_starttime=datetime(2022, 3, 19))
    c = g.get_substore('MDR', '00', 'HHZ')
    c.save(alarm.to_dataset())

    np.testing.assert_array_equal(c('alarm').values, alarm.values)
    g.starttime = datetime(2022, 3, 19)
    assert c('alarm').loc[:'2022-03-19T15:50'].isnull().all()


def test_xarray2netcdf_long_axis(tmp_path_factory):
    """
    An axis pre-allocated from 2000 stays small on disk, out-of-order writes
    land in place, and the axis stays evenly spaced as it grows.
    """
    import h5py

    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    g = Storage('test_experiment', rootdir=temp_dir, starttime=datetime(2000, 1, 1),
                endtime=datetime(2022, 3, 22), backend='netcdf',
                archive_starttime=datetime(2000, 1, 1))
    c = g.get_substore('MDR', '00', 'HHZ')

    def day(start, value):
        dates = pd.date_range(start, periods=144, freq='10min')
        return xr.DataArray(np.full(144, value), coords={'datetime': dates},
                            dims='datetime', name='rsam').to_dataset()

    c.save(day('2022-03-19', 1.))
    c.save(day('2022-03-21', 3.))  # leaves a day's gap
    c.save(day('2022-03-20', 2.))  # fills it out of order

    rsam = c('rsam')
    expected = pd.date_range('2000-01-01', '2022-03-21T23:50', freq='10min')
    assert rsam.sizes['datetime'] == len(expected)
    assert np.abs(rsam['datetime'].values - expected.values).max() < np.timedelta64(1, 's')
    assert rsam.loc[:'2022-03-18T23:50'].isnull().all()
    for date, value in [('2022-03-19', 1.), ('2022-03-20', 2.), ('2022-03-21', 3.)]:
        assert (rsam.loc[date] == value).all()

    path = c.feature_path('rsam')
    with h5py.File(path, 'r') as h5f:
        assert h5f['original/datetime'].compression == 'gzip'
    # an uncompressed axis of 1.17 million slots alone took 9.4 MB
    assert os.path.getsize(path) < 1e6


@pytest.mark.xfail(raises=OSError)
def test_xarray2netcdf_multi_access(tmp_path_factory):
    """
    Test writing xarray data to hdf5 while the file is open. This is currently
    not working with NetCDF4. See the following discussions for reference:
    https://github.com/pydata/xarray/issues/2887
    https://stackoverflow.com/questions/49701623/is-there-a-way-to-release-the-file-lock-for-a-xarray-dataset
    """
    temp_dir = tmp_path_factory.mktemp('test_xarray2netcdf')
    xdf1 = generate_test_data(
        dim=1, ndays=1, tstart=datetime(2022, 7, 18, 8, 0, 0))
    xdf2 = generate_test_data(
        dim=1, ndays=1, tstart=datetime(2022, 7, 19, 12, 0, 0))

    xarray2netcdf(xdf1, temp_dir)
    xdf_dummy = xr.open_dataset(os.path.join(temp_dir, 'rsam.nc'),
                                group='original', engine='h5netcdf')
    xarray2netcdf(xdf2, temp_dir)


@pytest.mark.slow
def test_netcdf_attribute_bug(tmp_path_factory):
    """
    Test to replicate behaviour when attribute is updated more than
    2^16 times.
    """
    temp_dir = tmp_path_factory.mktemp('test_netcdf_attribute_bug')
    g = Storage('test_experiment', rootdir=temp_dir, backend='netcdf')
    c = g.get_substore('MDR', '00', 'HHZ')
    tstart = datetime(2022, 7, 18, 8, 0, 0)
    for i in range(70000):
        if i % 1000 == 0:
            print(f'Iteration {i}')
        xdf = generate_test_data(tstart=tstart, dim=1, intervals=3, freq='1h')
        xdf.attrs['last_update'] = str(tstart + timedelta(hours=3))
        tstart += timedelta(days=1)
        c.save(xdf)
