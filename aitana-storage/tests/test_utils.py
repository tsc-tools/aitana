from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from aitana.storage.utils import (generate_test_data,
                         round_datetime,
                         floor_datetime,
                         get_dt,
                         snap_to_grid)


def test_generate_test_data():
    """
    Test data generation function.
    """
    tstart = datetime.now(timezone.utc) - timedelta(days=30)
    tstart = floor_datetime(tstart, timedelta(days=1))
    tstart = tstart.replace(tzinfo=None)
    data = generate_test_data(tstart='2023-01-01', freq='1min', seed=42,
                              ndays=3)
    assert 'datetime' in data.coords
    assert data.rsam.shape[0] == 3*24*60  # 24 hours + start point
    assert 'rsam' in data.data_vars
    assert 'dsar' in data.data_vars
    # Check for NaNs
    n_nans = np.isnan(data.dsar.values).sum()
    assert n_nans == 408


def test_floor_datetime_basic_10min():
    dt = datetime.fromisoformat("2025-11-27T10:12:43")
    out = floor_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 10, 0)


def test_floor_datetime_on_boundary():
    dt = datetime.fromisoformat("2025-11-27T10:20:00")
    out = floor_datetime(dt, 600)
    assert out == dt


def test_floor_datetime_timedelta_interval():
    dt = datetime.fromisoformat("2025-11-27T10:29:59")
    out = floor_datetime(dt, timedelta(minutes=10))
    assert out == datetime(2025, 11, 27, 10, 20, 0)


def test_floor_datetime_invalid_interval():
    dt = datetime.fromisoformat("2025-11-27T10:12:43")
    with pytest.raises(ValueError):
        floor_datetime(dt, 0)
    with pytest.raises(ValueError):
        floor_datetime(dt, -15)


def test_floor_datetime_preserves_timezone_utc():
    dt = datetime(2025, 11, 27, 10, 12, 43, tzinfo=timezone.utc)
    out = floor_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 10, 0, tzinfo=timezone.utc)


def test_floor_datetime_with_obspy_UTCDateTime():
    try:
        from obspy import UTCDateTime
    except Exception:
        pytest.skip("obspy not available")

    t = UTCDateTime(2025, 11, 27, 10, 12, 43)
    out = floor_datetime(t, 600)
    assert isinstance(out, UTCDateTime)
    assert out == UTCDateTime(2025, 11, 27, 10, 10, 0)


def test_round_datetime_basic_10min():
    dt = datetime.fromisoformat("2025-11-27T10:12:43")
    out = round_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 10)

    dt = datetime.fromisoformat("2025-11-27T10:10:00")
    out = round_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 10)

    dt = datetime.fromisoformat("2025-11-27T10:17:00")
    out = round_datetime(dt, 600)
    assert out == datetime(2025, 11, 27, 10, 20)


# Millisecond offsets, like traces that start a few milliseconds late
JITTER = np.array([4, -3, 2, 5, -5, 1, 0, -2, 3, -4]).astype('timedelta64[ms]')


def test_get_dt():
    grid = pd.date_range('2023-01-01', periods=10, freq='10min').values
    assert get_dt(grid) == pd.Timedelta('10min')
    assert get_dt(grid + JITTER) == pd.Timedelta('10min')
    # float hours decode up to a microsecond off
    assert get_dt(grid + np.timedelta64(-256, 'ns')) == pd.Timedelta('10min')
    fast = pd.date_range('2023-01-01', periods=10, freq='250ms').values
    assert get_dt(fast) == pd.Timedelta('250ms')
    with pytest.raises(ValueError, match="two time stamps"):
        get_dt(grid[:1])


def test_snap_to_grid():
    # the grid keeps the phase of the data, here window centres at :05
    grid = pd.date_range('2023-01-01 00:05', periods=10, freq='10min').values
    np.testing.assert_array_equal(snap_to_grid(grid + JITTER, pd.Timedelta('10min')), grid)
    # an explicit origin is any point of the grid
    origin = np.datetime64('2000-01-01T00:05')
    np.testing.assert_array_equal(
        snap_to_grid(grid + JITTER, pd.Timedelta('10min'), origin), grid)
    assert snap_to_grid(grid, pd.Timedelta('10min')).dtype == np.dtype('datetime64[ns]')


def test_snap_to_grid_check():
    grid = pd.date_range('2023-01-01', periods=10, freq='10s').values
    # jitter is fine
    snap_to_grid(grid + JITTER, pd.Timedelta('10s'), check=True)
    # a different phase is not: window centres at :05 on a grid at :00
    with pytest.raises(ValueError, match="off the grid"):
        snap_to_grid(grid + np.timedelta64(5, 's'), pd.Timedelta('10s'),
                     origin=np.datetime64('2000-01-01'), check=True)
    # nor are time stamps that mix phases
    mixed = np.concatenate([grid[:5], grid[5:] + np.timedelta64(3, 's')])
    with pytest.raises(ValueError, match="off the grid"):
        snap_to_grid(mixed, pd.Timedelta('10s'), check=True)
