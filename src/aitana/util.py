from datetime import datetime, timezone
import os
import pandas as pd
from functools import wraps
import hashlib
import logging

import numpy as np

logger = logging.getLogger(__name__)


def generate_cache_filename(func_name, args, kwargs):
    """
    Generate a unique filename for caching based on function name and arguments.
    """
    # Serialize arguments to a string representation
    instance = args[0] if args and hasattr(
        args[0], '__class__') else None

    if instance:
        args_repr = "_".join(repr(arg) for arg in args[1:])
    else:
        args_repr = "_".join(repr(arg) for arg in args)
    kwargs_repr = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

    # Combine everything into a single string
    combined = f"{func_name}_{args_repr}_{kwargs_repr}"

    # Hash the combined string to ensure the filename is valid and not too long
    hash_object = hashlib.md5(combined.encode("utf-8"))
    hashed_name = hash_object.hexdigest()

    # Construct the filename
    return f"{func_name}_{hashed_name}"  # Use .pkl for cached dataframes


def cache_dataframe(cache_dir: str = ""):
    """
    Decorator to cache pandas DataFrames, handle date ranges, and persist the cache to disk.

    Args:
        cache_key (str): The attribute name for the cached DataFrame.
        use_class_attrs (bool): If True, fetch start_date and end_date from the class instance if available.
        cache_file (str): Filepath for persisting the cache.
    """
    if not cache_dir:
        cache_dir = os.path.join(os.environ['HOME'], ".aitana_cache")
        os.makedirs(cache_dir, exist_ok=True)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, start_date=None, end_date=None, clear_cache=False, **kwargs):
            cache_key = generate_cache_filename(func.__name__, args, kwargs)
            cache_file = os.path.join(cache_dir, f"{cache_key}.csv")
            if clear_cache:
                if os.path.exists(cache_file):
                    os.remove(cache_file)

            # Check if the function is a method and should use class attributes
            instance = args[0] if args and hasattr(
                args[0], '__class__') else None

            if instance is not None:
                # Use instance attributes if not explicitly provided
                end_date = end_date or getattr(instance, 'end_date', None)
                start_date = start_date or getattr(
                    instance, 'start_date', None)

            if end_date is None:
                raise ValueError("end_date must be provided.")

            end_date = pd.to_datetime(end_date, utc=True)
            start_date = pd.to_datetime(start_date, utc=True)

            # Load the cache from disk if it exists
            if not hasattr(wrapper, 'cached_df'):
                if os.path.exists(cache_file):
                    logger.debug(f"Loading cache from {cache_file}")
                    cached_df = pd.read_csv(
                        cache_file, parse_dates=True, index_col=0,
                        date_format="ISO8601")
                else:
                    cached_df = pd.DataFrame()
                setattr(wrapper, 'cached_df', cached_df)
            else:
                cached_df = getattr(wrapper, 'cached_df')

            # Check the date range of the cached data
            if not cached_df.empty:
                cached_end = cached_df.index.max()
            else:
                cached_end = None

            # Determine missing ranges
            missing_end = end_date if cached_end is None or end_date > cached_end else None

            # If there are missing ranges, fetch the data
            if missing_end:
                logger.debug("Fetching missing data...")
                if instance is not None:
                    setattr(instance, 'end_date', missing_end)
                    missing_data = func(*args, **kwargs)
                else:
                    try:
                        missing_data = func(*args, end_date=end_date, **kwargs)
                    except TypeError:
                        missing_data = func(*args, **kwargs)

                if not isinstance(missing_data, pd.DataFrame):
                    raise ValueError(
                        "The decorated function must return a pandas DataFrame.")

                if not missing_data.empty:
                    missing_data.index = pd.to_datetime(
                        missing_data.index)  # Ensure the index is datetime

                # Update the cache
                cached_df = missing_data
                setattr(wrapper, 'cached_df', cached_df)

                # Persist the updated cache to disk
                logger.debug(f"Saving cache to {cache_file}")
                cached_df.to_csv(cache_file)
            # Return the relevant slice of the cached DataFrame
            return_df = cached_df.loc[start_date:end_date]
            if return_df.empty:
                raise ValueError("No data available for the given date range.")
            return return_df

        return wrapper
    return decorator


def gradient(df, period="14D"):
    """
    Compute gradient for time series by first smoothing
    the time series and then computing the first-order difference.

    Parameters:
    -----------
        :param df: Dataframe
        :type df: :class:`~pandas.DataFrame`
        :param period: Period over which to compute a rolling mean.
        :type period: str
    """
    df_smooth = df.rolling(period).mean()
    df_grad = df_smooth.diff()
    df_grad.loc[df_grad.index[0], "obs"] = 0.0
    return df_grad


def eqRate(cat, fixed_time=None, fixed_nevents=None, enddate=datetime.utcnow()):
    """
    Compute earthquake rate.

    :param cat: A catalogue of earthquakes as returned by
                :method:`pyvolprob.load_ruapehu_earthquakes`
    :type cat: :class:`pandas.DataFrame`
    :param fixed_time: If not 'None', compute the earthquake rate
                       based on a fixed-length time window given in
                       days.
    :type fixed_time: int
    :param fixed_nevents: If not None, compute the earthquake rate
                          based on a fixed number of events.
    :type fixed_nevents: int
    :param enddate: The latest date of the time-series.
                    Mainly needed for testing.
    :type enddate: :class:`datetime.datetime`

    """
    if fixed_time is not None and fixed_nevents is not None:
        raise ValueError(
            "Please define either 'fixed_time' or 'fixed_nevents'")

    dates = cat.index.values
    if fixed_time is not None:
        ds = pd.Series(np.ones(len(dates)), index=dates)
        ds = pd.concat([ds, pd.Series([np.nan], index=[enddate])])
        ds.sort_index(inplace=True)
        ds = ds.rolling("{:d}D".format(fixed_time)).count() / fixed_time
        ds.index -= pd.Timedelta("{:d}D".format(int(fixed_time / 2.0)))
        return pd.DataFrame({"obs": ds}).tz_localize('utc')
    elif fixed_nevents is not None:
        nevents = dates.shape[0]
        aBin = np.zeros(nevents - fixed_nevents, dtype="datetime64[ns]")
        aRate = np.zeros(nevents - fixed_nevents)
        iS = 0
        for s in np.arange(fixed_nevents, nevents):
            i1, i2 = s - fixed_nevents, s
            dt = (dates[i2] - dates[i1]).astype("timedelta64[s]")
            dt_days = dt.astype(float) / 86400.0
            aBin[iS] = dates[i1] + 0.5 * dt
            aRate[iS] = fixed_nevents / dt_days
            iS += 1
        return pd.DataFrame({"obs": aRate}, index=aBin).tz_localize('utc')
    else:
        raise ValueError(
            "Please define either 'fixed_time' or 'fixed_nevents'")


def reindex(df, dates, fill_method=None, ffill_interval=14):
    """
    Reindex and forward fill to generate a
    timeseries that can be used to set the evidence
    for a BN.

    :param df: Dataframe to reindex
    :type df: :class:`pandas.DataFrame`
    :param dates: new date index
    :type dates: :class:`pandas.DataTimeIndex`
    :param ffill_interval: Best-by interval for data
    :type ffill_interval: int
    """
    if fill_method is None:
        return df["obs"].resample("D").max().reindex(dates)
    elif fill_method == "ffill":
        return df["obs"].reindex(dates, method="ffill", limit=ffill_interval)
    elif fill_method == "interpolate":
        df_tmp = df["obs"].resample("D").max().reindex(dates)
        return df_tmp.interpolate(method="linear")
    else:
        msg = "'fill_method' has to be one of "
        msg += "[None, 'ffill', 'interpolate']"
        raise ValueError(msg)
