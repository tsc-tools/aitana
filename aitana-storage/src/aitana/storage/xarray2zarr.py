from datetime import datetime, timezone
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
try:
    from zarr.errors import PathNotFoundError
except ImportError:
    class PathNotFoundError(Exception):
        pass

from .utils import merge_arrays, fill_time_gaps, get_dt, snap_to_grid

logger = logging.getLogger(__name__)


def _init_timeseries_store(path: str, start: np.datetime64, stop: np.datetime64, interval: pd.Timedelta,
                           data_vars: dict, group: str = "original", chunk_size: int = 10,
                           timedim: str = "datetime") -> xr.DataArray:
    """
    Initialize an empty zarr store for time series data. This facilitates writing data out
    of sequence and avoid prepending which is costly and difficult to get right.

    Parameters
    ----------
    path : str
        Path to the zarr store.
    start : np.datetime64
        Start time of the zarr store.
    stop : np.datetime64
        End time of the zarr store.
    interval : pd.Timedelta
        Sampling interval string (e.g. '1H', '15T') for the time dimension
    data_vars : dict
        Dictionary defining the data variables to create. Keys are variable names,
        values are tuples of (dims, shape, dtype) where dims is a tuple of dimension
        names (excluding the time dimension), shape is a tuple of dimension sizes
        (excluding the time dimension), and dtype is the numpy data type.
    group : str, optional
        Group name in the zarr store, by default "original"
    chunk_size : int, optional
        Chunk size in number of time steps, by default 10
    timedim : str, optional
        Name of the time dimension, by default "datetime"

    """
    # Make the zarr store a multiple of chunk_size
    stop_ts = pd.Timestamp(stop)
    start_ts = pd.Timestamp(start)
    chunk_length = int(chunk_size)
    if chunk_length <= 0:
        raise ValueError("chunk_size must be a positive integer")
    total_steps = int((stop_ts - start_ts) // interval) + 1
    if total_steps < 1:
        total_steps = chunk_length
    if total_steps % chunk_length:
        required_steps = ((total_steps + chunk_length - 1) //
                          chunk_length) * chunk_length
        start_ts = stop_ts - interval * (required_steps - 1)
    time_index = pd.date_range(start=start_ts, end=stop_ts, freq=interval)
    ds = xr.Dataset()
    name, value = list(data_vars.items())[0]
    dims, coords, shape, dtype = value
    dims = dims + (timedim,)
    shape = tuple(shape) + (len(time_index),)
    # Create coordinates for gap dataset
    new_coords = {timedim: time_index}
    for coord_name, coord in coords.items():
        if coord_name != timedim:
            new_coords[coord_name] = coord

    xda = xr.DataArray(
        np.full(shape, np.nan, dtype=dtype),
        coords=new_coords,
        dims=dims,
        name=name
    )
    xda = xda.chunk(
        {timedim: chunk_size, **{d: -1 for d in dims[:-1]}})
    xda.to_zarr(path, group=group, mode="w")
    return xda


def _fill_time_gaps_between_datasets(xds_existing: xr.DataArray, xds_new: xr.DataArray, interval: pd.Timedelta,
                                     timedim: str = 'datetime', chunk_size: int = 10) -> xr.DataArray:
    """
    Fill gaps between existing and new datasets.

    Parameters
    ----------
    xds_existing : xr.Dataset
        Existing dataset on disk
    xds_new : xr.Dataset
        New dataset to append
    timedim : str
        Name of the time dimension

    Returns
    -------
    xr.Dataset
        Combined dataset with gaps filled
    """

    existing_endpoint = xds_existing[timedim].values
    # Get time ranges
    gap_start = existing_endpoint + interval
    gap_end = xds_new[timedim].values[0] - interval

    # Prepare shape for gap filling; the time axis can be anywhere, e.g.
    # (frequency, datetime) for spectra, (datetime, channel, frequency) for
    # scattering coefficients
    shape_list = list(xds_new.shape)
    time_axis = list(xds_new.dims).index(timedim)

    if gap_start <= gap_end:
        gap_times = pd.date_range(start=gap_start, end=gap_end, freq=interval)

        # Create NaN array with same shape as variable but for gap times
        gap_shape = list(shape_list)
        gap_shape[time_axis] = len(gap_times)
        gap_values = np.full(gap_shape, np.nan, dtype=xds_new.dtype)

        # Create coordinates for gap dataset
        gap_coords = {timedim: gap_times}
        for coord_name, coord in xds_new.coords.items():
            if coord_name != timedim:
                gap_coords[coord_name] = coord

        gap_data = xr.DataArray(
            gap_values,
            coords=gap_coords,
            dims=xds_new.dims,
            name=xds_new.name
        )

        # Combine: existing + gap + new
        combined = xr.concat([gap_data, xds_new], dim=timedim)
    else:
        combined = xds_new

    # ensure new array aligns with chunk size
    arr_len = combined.sizes[timedim]
    need = -arr_len % chunk_size  # 0..chunklen-1
    if need > 0:
        start = combined[timedim].values[-1] + interval
        pad_times = pd.date_range(start=start, periods=need, freq=interval)
        pad_shape = list(shape_list)
        pad_shape[time_axis] = len(pad_times)
        pad_vals = np.full(pad_shape, np.nan, dtype=xds_new.dtype)
        pad_coords = {timedim: pad_times}
        for coord_name, coord in xds_new.coords.items():
            if coord_name != timedim:
                pad_coords[coord_name] = coord
        pad_da = xr.DataArray(pad_vals, coords=pad_coords,
                              dims=xds_new.dims,
                              name=xds_new.name)
        combined = xr.concat([combined, pad_da], dim=timedim)

    return combined


def _update_meta_data(fout: str,
                      last_datapoint: np.datetime64,
                      resolution: float | None = None,
                      meta_group: str = "meta") -> None:
    """
    Append current update time (and last_datapoint) to meta group.

    Parameters
    ----------
    fout : str
        Base zarr store path (per-variable .zarr directory).
    last_datapoint : np.datetime64
        Latest data time in the feature.
    resolution : float | None
        Optional time resolution (hours) to store once.
    meta_group : str
        Group name for metadata.
    """

    now = np.datetime64(datetime.now(tz=timezone.utc), 's')
    new_update = xr.DataArray([now],
                              coords={'update': [now]},
                              dims=['update'],
                              name='update_log')
    new_last = xr.DataArray([last_datapoint],
                            coords={'endtime': [now]},
                            dims=['endtime'],
                            name='last_datapoint')

    try:
        meta = xr.open_zarr(fout, group=meta_group, chunks=None)
        # Existing vars -> concatenate
        update_old = meta.get('update_log')
        last_old = meta.get('last_datapoint')
        res_da_old = meta.get('resolution').values[()]
        new_update = xr.concat([update_old, new_update], dim='update')
        new_last = xr.concat([last_old, new_last], dim='endtime')
        if abs(resolution - res_da_old) > 1e-5:
            raise ValueError(f"Resolution mismatch for {fout}: "
                             f"{res_da_old} != {resolution}")
        res_da = xr.DataArray(resolution, name='resolution')
    except Exception:
        # First creation
        res_da = xr.DataArray(
            resolution, name='resolution') if resolution is not None else None

    vars = {'update_log': new_update, 'last_datapoint': new_last}
    if res_da is not None:
        vars['resolution'] = res_da
    xr.Dataset(vars).to_zarr(fout, group=meta_group, mode='w')


def xarray2zarr(xds: xr.Dataset, path: str, group='original',
                chunk_size: int = 1000, timedim: str = 'datetime', interval: str = None,
                archive_starttime: datetime = datetime(2000, 1, 1)) -> None:
    """
    Write xarray dataset to zarr files.

    Parameters
    ----------
    xds : xr.Dataset
        Dataset to write.
    path : str
        Path to write the dataset.
    mode : str, optional
        Write mode, by default 'a'.
    group : str, optional
        Group name, by default 'original'
    chunk_size : int, optional
        Chunk size as the number of days.
    timedim : str
        Name of the time dimension, by default 'datetime'
    interval : str, optional
        Sampling interval. If None, it is inferred from the data, and time
        stamps further off the store's grid than jitter raise a ValueError.
        If given, data go to the nearest grid point.
    archive_starttime : datetime, optional
        Start of a new store; its grid runs through the data's time stamps.
    fill_gaps : bool, optional
        Whether to fill time gaps with NaN before writing, by default False

    Returns
    -------
    None
    """

    if timedim not in xds.dims:
        raise ValueError(f"{timedim} dimension not found in Dataset.")

    check_grid = interval is None
    if interval is None:
        interval = get_dt(np.unique(xds[timedim].values))
    else:
        interval = pd.to_timedelta(interval)

    for feature in xds.data_vars.keys():
        fout = os.path.join(path, feature + '.zarr')
        try:
            xds_existing = xr.open_zarr(fout, group=group)
            has_store = True
        except (PathNotFoundError, FileNotFoundError, KeyError):
            has_store = False

        # Put the time stamps on the store's grid (a new store takes the data's
        # own), as they're matched to it exactly below; then fill gaps
        origin = xds_existing[timedim].values[0] if has_store else None
        xda = xds[feature].assign_coords(
            {timedim: snap_to_grid(xds[timedim].values, interval, origin,
                                   check=check_grid)})
        xda = xda.drop_duplicates(timedim, keep='last')
        xda = fill_time_gaps(xda.to_dataset(), timedim=timedim)[feature]

        last_dp = xda[timedim].values[-1]
        _update_meta_data(fout, last_dp, resolution=float(
            interval / pd.Timedelta(1, 'h')))

        if not has_store:
            logger.debug("Creating new zarr store.")
            shape_list = list(xda.shape)
            dims_list = list(xda.dims)
            shape_list.pop(dims_list.index(timedim))
            dims_list.pop(dims_list.index(timedim))
            xds_existing = _init_timeseries_store(
                fout,
                start=np.datetime64(archive_starttime),
                stop=xda[timedim].values[-1],
                interval=interval,
                data_vars={
                    feature: (tuple(dims_list), xda.coords,
                              tuple(shape_list), xda.dtype)},
                group=group,
                chunk_size=chunk_size,
                timedim=timedim
            )

        if xds_existing[timedim][0] > xda[timedim][0]:
            raise ValueError("New data ends before existing data starts. "
                             "Prepending to existing data is currently not supported.")

        elif xds_existing[timedim][-1] < xda[timedim][0]:
            logger.debug("Appending data to existing zarr store.")
            xda_new = _fill_time_gaps_between_datasets(xds_existing[feature].isel({timedim: -1}),
                                                       xda, interval, chunk_size=chunk_size)
            xda_new.to_zarr(fout, group=group, mode='a',
                            append_dim=timedim)
        else:
            logger.debug("Data in zarr store overlaps with new data.")
            logger.debug(
                f"Endtime of existing data: {xds_existing[timedim][-1].values}")
            logger.debug(f"Starttime of new data: {xda[timedim][0].values}")
            existing_times = xds_existing[timedim].values
            new_times = xda[timedim].values

            overlap_times, idx_existing, idx_new = np.intersect1d(
                existing_times,
                new_times,
                assume_unique=True,
                return_indices=True,
            )
            region = {}
            for dim in xda.dims:
                if dim == timedim:
                    start = int(idx_existing.min())
                    stop = start + len(idx_existing)
                    region[dim] = slice(start, stop)
                else:
                    region[dim] = 'auto'
            xda.isel({timedim: idx_new}).to_zarr(
                fout, group=group, mode='r+', region=region)
            remainder = xda.drop_sel({timedim: new_times[idx_new]})
            if remainder.sizes[timedim] > 0:
                xda_new = _fill_time_gaps_between_datasets(xds_existing[feature].isel({timedim: -1}),
                                                           remainder, interval, chunk_size=chunk_size)
                xda_new.to_zarr(fout, group=group, mode='a',
                                append_dim=timedim)
