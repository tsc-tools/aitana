from typing import Union
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import xarray as xr


def generate_test_data(dim=1, ndays=30, nfreqs=10,
                       tstart=datetime.now(),
                       freq='10min', intervals=None,
                       feature_names=None, seed=42,
                       freq_names=None, add_nans=True):
    """
    Generate a 1D or 2D feature for testing.

    Parameters
    ----------
    dim : int
        Dimension of the data (1 or 2).
    ndays : int
        Number of days to generate data for.
    nfreqs : int
        Number of frequencies (only for dim=2).
    tstart : datetime
        Start time of the data.
    freq : str
        Frequency of the data (e.g., '10min').
    intervals : int
        Number of intervals to generate. If None, calculated from ndays and freq.
    feature_names : list
        Names of the features to generate.
    seed : int
        Random seed for reproducibility.
    freq_names : list
        Names of the frequency dimensions (only for dim=2).
    add_nans : bool
        Whether to add NaN values to the data.

    Returns
    -------
    xr.Dataset
        Generated test dataset.
    """
    assert dim < 3
    assert dim > 0

    if intervals is None:
        nints = ndays * int(pd.Timedelta('1h')/pd.Timedelta(freq)) * 24
    else:
        nints = intervals
    dates = pd.date_range(tstart, freq=freq, periods=nints)
    rs = np.random.default_rng(seed)
    # Random walk as test signal
    data = np.abs(np.cumsum(rs.normal(0, 8., len(dates))))
    if dim == 2:
        data = np.tile(data, (nfreqs, 1))
    # Add 10% NaNs
    idx_nan = rs.integers(0, nints-1, int(0.1*nints))

    xds_dict = {}
    if dim == 1:
        if add_nans:
            data[idx_nan] = np.nan
        if feature_names is None:
            feature_names = ['rsam', 'dsar']
        for feature in feature_names:
            xds_dict[feature] = xr.DataArray(
                data, coords=[dates], dims=['datetime'])
    if dim == 2:
        if add_nans:
            data[:, idx_nan] = np.nan
        freqs = np.arange(nfreqs)
        if feature_names is None:
            feature_names = ['ssam', 'filterbank']
        if freq_names is None:
            freq_names = ['frequency', 'fbfrequency']

        for feature_name, freq_name in zip(feature_names, freq_names):
            xds_dict[feature_name] = xr.DataArray(
                data, coords=[freqs, dates], dims=[freq_name, 'datetime'])
    xds = xr.Dataset(xds_dict)
    xds.attrs['starttime'] = dates[0].isoformat()
    xds.attrs['endtime'] = dates[-1].isoformat()
    xds.attrs['station'] = 'MDR'
    xds.attrs['interval'] = '10min'
    return xds


def round_datetime(dt: datetime, interval: Union[int, float, timedelta]) -> datetime:
    """
    Find closest multiple of interval to given time.

    Parameters:
    -----------
    dt : datetime
        The datetime to round.
    interval : Union[int, float, timedelta]
        The interval to which to round the datetime.

    Returns:
    --------
    datetime
        The rounded datetime.
    """
    # Normalize interval to whole seconds (supports float/timedelta inputs)
    if isinstance(interval, timedelta):
        interval_sec = int(interval.total_seconds())
    else:
        interval_sec = int(interval)

    if interval_sec <= 0:
        raise ValueError("interval must be positive (seconds)")

    # Accept ObsPy UTCDateTime transparently (preserve type on return)
    _is_obspy = False
    try:
        from obspy import UTCDateTime as _UTCDateTime  # type: ignore
        if isinstance(dt, _UTCDateTime):
            _is_obspy = True
            dt_py = dt.datetime  # Python datetime in UTC
        else:
            dt_py = dt
    except Exception:
        dt_py = dt

    epoch = (
        datetime(1970, 1, 1)
        if dt_py.tzinfo is None
        else datetime(1970, 1, 1, tzinfo=dt_py.tzinfo)
    )

    # Compute integer seconds since epoch to avoid float precision issues
    seconds = int((dt_py - epoch).total_seconds())
    floored = (seconds + 0.5 * interval_sec) % interval_sec
    rounded = epoch + timedelta(seconds=seconds + 0.5 * interval_sec - floored)

    if _is_obspy:
        from obspy import UTCDateTime as _UTCDateTime  # type: ignore
        return _UTCDateTime(rounded)

    return rounded


def floor_datetime(dt: datetime, interval: Union[int, float, timedelta]) -> datetime:
    """
    Floor a datetime to the latest multiple of a given interval.

    Assumes ``dt`` represents a UTC time (naive or tz-aware is fine) and
    aligns against the Unix epoch 1970-01-01T00:00:00Z. The interval is in
    seconds (int/float) or a timedelta. Returns a datetime with the same
    "naive vs aware" form as ``dt``.

    Examples
    --------
    >>> from datetime import datetime
    >>> floor_datetime(datetime.fromisoformat('2025-11-27T10:12:43'), 600)
    datetime.datetime(2025, 11, 27, 10, 10)
    """

    # Normalize interval to whole seconds (supports float/timedelta inputs)
    if isinstance(interval, timedelta):
        interval_sec = int(interval.total_seconds())
    else:
        interval_sec = int(interval)

    if interval_sec <= 0:
        raise ValueError("interval must be positive (seconds)")

    # Accept ObsPy UTCDateTime transparently (preserve type on return)
    _is_obspy = False
    try:
        from obspy import UTCDateTime as _UTCDateTime  # type: ignore
        if isinstance(dt, _UTCDateTime):
            _is_obspy = True
            dt_py = dt.datetime  # Python datetime in UTC
        else:
            dt_py = dt
    except Exception:
        dt_py = dt

    epoch = (
        datetime(1970, 1, 1)
        if dt_py.tzinfo is None
        else datetime(1970, 1, 1, tzinfo=dt_py.tzinfo)
    )

    # Compute integer seconds since epoch to avoid float precision issues
    seconds = int((dt_py - epoch).total_seconds())
    floored = seconds - (seconds % interval_sec)
    rounded = epoch + timedelta(seconds=floored)

    if _is_obspy:
        from obspy import UTCDateTime as _UTCDateTime  # type: ignore
        return _UTCDateTime(rounded)

    return rounded


def _precision(spacing):
    """
    Unit that time stamps on a grid with this spacing are rounded to.

    Parameters
    ----------
    spacing : pd.Timedelta
        Grid spacing.

    Returns
    -------
    str
        'ms' for spacings below a second, otherwise 's'.
    """
    return 's' if pd.Timedelta(spacing) >= pd.Timedelta(1, 's') else 'ms'


def get_dt(times):
    """
    Infer the sampling interval of a time axis.

    The median spacing, rounded to whole seconds (whole milliseconds below a
    second), so that time stamps slightly off the grid don't change the
    interval: times stored as float hours decode up to a microsecond off,
    and traces can start a few milliseconds late.

    Parameters
    ----------
    times : array-like of datetime64
        At least two sorted time stamps.

    Returns
    -------
    pd.Timedelta
        The sampling interval.
    """
    values = np.asarray(times, dtype='datetime64[ns]')
    if len(values) < 2:
        raise ValueError(
            "Need at least two time stamps to infer the sampling interval.")
    spacing = pd.to_timedelta(np.diff(values)).median()
    return spacing.round(_precision(spacing))


def snap_to_grid(times, interval, origin=None, check=False):
    """
    Round time stamps to the nearest point of a regular grid.

    Parameters
    ----------
    times : array-like of datetime64
        Time stamps to snap.
    interval : pd.Timedelta
        Grid spacing.
    origin : datetime64-like, optional
        A point of the grid. By default the first time stamp, rounded to whole
        seconds (whole milliseconds for spacings below a second), so that the
        grid keeps the phase of the data, e.g. time stamps at window centres.
    check : bool, optional
        Raise a ValueError if a time stamp is further from its grid point than
        jitter explains: 1% of the interval, at most a second.

    Returns
    -------
    np.ndarray
        datetime64[ns] time stamps ``origin + k * interval``; halfway cases
        round up.
    """
    values = np.asarray(times, dtype='datetime64[ns]')
    step = pd.Timedelta(interval).value
    if origin is None:
        origin = pd.Timestamp(values[0]).round(_precision(interval))
    origin = np.datetime64(pd.Timestamp(origin).to_datetime64(), 'ns')
    # integer nanoseconds, so that the grid points are exact
    offset = (values - origin).astype(np.int64)
    k = (offset + step // 2) // step
    snapped = origin + (k * step).astype('timedelta64[ns]')
    if check:
        tolerance = min(pd.Timedelta(interval) / 100, pd.Timedelta(1, 's'))
        off = np.abs((values - snapped).astype(np.int64)).max()
        if off > tolerance.value:
            raise ValueError(
                f"Time stamps are up to {pd.Timedelta(off, 'ns')} off the grid "
                f"{pd.Timestamp(origin)} + k * {pd.Timedelta(interval)}, more than "
                f"the {tolerance} allowed for jitter. Data with a different phase "
                "(e.g. stamped at window centres instead of starts) can't share "
                "an archive.")
    return snapped


def fill_time_gaps(xds: xr.Dataset, timedim: str = 'datetime') -> xr.Dataset:
    """
    Fill gaps in time series with NaN values by reindexing to a complete datetime range.

    Parameters
    ----------
    xds : xr.Dataset
        Input dataset with potential time gaps
    freq : str, optional
        Frequency string (e.g., 'H', 'D', '15min'). If None, will try to infer.
    timedim : str
        Name of the time dimension, by default 'datetime'

    Returns
    -------
    xr.Dataset
        Dataset with gaps filled with NaN
    """
    if timedim not in xds.coords:
        raise ValueError(
            f"{timedim} coordinate not found in dataset coordinates.")

    # Infer sample interval
    dt = get_dt(xds.coords[timedim])
    start_time = xds[timedim].values[0]
    end_time = xds[timedim].values[-1]
    complete_time = pd.date_range(start=start_time, end=end_time, freq=dt)

    # Reindex to fill gaps with NaN
    return xds.reindex({timedim: complete_time})


def merge_arrays(xds_old: xr.DataArray, xds_new: xr.DataArray,
                 timedim: str = 'datetime', resolution: float = None) -> xr.DataArray:
    """
    Merge two xarray datasets with the same datetime index.

    Parameters
    ----------
    xds_old : xr.DataArray
        Old array.
    xds_new : xr.DataArray
        New array.
    resolution : float
        Time resolution in hours.

    Returns
    -------
    xr.DataArray
        Merged array.
    """
    xda_old = xds_old.drop_duplicates(
        timedim, keep='last')
    xda_new = xds_new.drop_duplicates(
        timedim, keep='last')
    xda_new = xda_new.combine_first(xda_old)
    if resolution is not None:
        new_dates = pd.date_range(
            xda_new[timedim].values[0],
            xda_new[timedim].values[-1],
            freq=f'{resolution}h')
        xda_new = xda_new.reindex(dict(timedim=new_dates))
    xda_new = fill_time_gaps(xda_new, timedim=timedim)
    return xda_new


def main():
    from .storage import Storage
    import logging

    logger = logging.getLogger(__name__)
    logger.info("Generating test data")
    rootdir = '/tmp'
    g = Storage('volcanoes', rootdir=rootdir)
    st1 = g.get_substore('Mt Doom', 'MDR', '00', 'BHZ')
    st2 = g.get_substore('Misty Mountain', 'MMS', '10', 'HHZ')

    # Get start time of the current 10 minute window
    tstart = datetime.now(timezone.utc).timestamp()
    tstart -= tstart % 600
    tstart = datetime.fromtimestamp(tstart, timezone.utc)
    tstart -= timedelta(days=30)
    tstart = tstart.replace(tzinfo=None)
    logger.info(f"Start time: {tstart}")

    # Generate test data
    xdf_1D = generate_test_data(dim=1, tstart=tstart, add_nans=False, seed=42)
    xdf_1D_1 = generate_test_data(
        dim=1, tstart=tstart, add_nans=False, seed=24)
    xdf_2D = generate_test_data(
        dim=2, tstart=tstart, add_nans=False, seed=1234)
    xdf_2D_1 = generate_test_data(
        dim=2, tstart=tstart, add_nans=False, seed=4321)
    logger.info("Saving test data to " + st1.path)
    st1.save(xdf_1D)
    st1.save(xdf_2D)
    logger.info("Saving test data to " + st2.path)
    st2.save(xdf_1D_1)
    st2.save(xdf_2D_1)

    # Generate the data used by the examples in package_data/index.html
    tg_start = datetime(2023, 1, 1)
    tg = Storage('testgroup', rootdir=rootdir, archive_starttime=tg_start)
    xdf_tg = generate_test_data(
        dim=1, ndays=10, tstart=tg_start, add_nans=False, seed=42)
    logger.info("Saving test data to " + tg.path)
    tg.save(xdf_tg)
    tg.get_substore('WIZ', '00', 'HHZ').save(xdf_tg)


if __name__ == '__main__':
    main()
