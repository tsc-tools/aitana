import hashlib
import inspect
import math as M
import os
from datetime import datetime
from functools import wraps
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Trace, UTCDateTime
from scipy.signal import chirp, sweep_poly

from aitana.logging_config import get_logger

logger = get_logger(__name__)


def get_defaults_with_names(func):
    """
    Get parameter names and their default values
    from a function object.
    """
    sig = inspect.signature(func)
    defaults = func.__defaults__ or ()
    kwdefaults = func.__kwdefaults__ or {}

    # Positional/keyword arguments with defaults
    param_names = list(sig.parameters.keys())
    positional_defaults = param_names[-len(defaults) :] if defaults else []
    positional_defaults_with_names = dict(zip(positional_defaults, defaults))

    # Combine with keyword-only defaults
    all_defaults = {**positional_defaults_with_names, **kwdefaults}
    return all_defaults


def generate_cache_key(func_name, args, kwargs):
    """
    Generate a unique filename for caching based on function name and arguments.
    """
    # Serialize arguments to a string representation
    instance = args[0] if args and hasattr(args[0], "__class__") else None

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


def _is_class_method(func):
    """
    Check if a function is a class method.
    """
    func_params = list(inspect.signature(func).parameters)
    return func_params and func_params[0] == "self"


def cache_dataframe(cache_dir: str = ""):
    """
    Decorator to cache pandas DataFrames, handle date ranges, and persist the cache to disk.

    Args:
        cache_key (str): The attribute name for the cached DataFrame.
        use_class_attrs (bool): If True, fetch start_date and end_date from the class instance if available.
        cache_file (str): Filepath for persisting the cache.
    """
    if not cache_dir:
        cache_dir = os.path.join(Path.home(), ".aitana_cache")
        os.makedirs(cache_dir, exist_ok=True)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, start_date=None, end_date=None, clear_cache=False, **kwargs):
            cache_key = generate_cache_key(func.__name__, args, kwargs)
            cache_file = os.path.join(cache_dir, f"{cache_key}.csv")
            if clear_cache:
                if os.path.exists(cache_file):
                    os.remove(cache_file)

            instance = args[0] if _is_class_method(func) else None

            if instance is not None:
                # Use instance attributes if not explicitly provided
                end_date = end_date or getattr(instance, "end_date", None)
                start_date = start_date or getattr(instance, "start_date", None)

            if end_date is None:
                default_params = get_defaults_with_names(func)
                if "end_date" in default_params:
                    end_date = default_params["end_date"]
                else:
                    raise ValueError("end_date must be provided.")

            end_date = pd.to_datetime(end_date, utc=True)
            if start_date is not None:
                start_date = pd.to_datetime(start_date, utc=True)

            # Load the cache from disk if it exists
            if not hasattr(wrapper, f"cached_df_{cache_key}"):
                if os.path.exists(cache_file):
                    logger.debug(f"Loading cache from {cache_file}")
                    cached_df = pd.read_csv(
                        cache_file, parse_dates=True, index_col=0, date_format="ISO8601"
                    )
                else:
                    cached_df = pd.DataFrame()
                setattr(wrapper, f"cached_df_{cache_key}", cached_df)
            else:
                cached_df = getattr(wrapper, f"cached_df_{cache_key}")

            # Check the date range of the cached data
            if not cached_df.empty:
                cached_end = cached_df.index.max()
            else:
                cached_end = None

            # Determine missing ranges
            missing_end = (
                end_date if cached_end is None or end_date > cached_end else None
            )

            # If there are missing ranges, fetch the data
            if missing_end:
                logger.debug("Fetching missing data...")
                if instance is not None:
                    setattr(instance, "end_date", missing_end)
                    missing_data = func(*args, **kwargs)
                else:
                    try:
                        missing_data = func(*args, end_date=end_date, **kwargs)
                    except TypeError:
                        missing_data = func(*args, **kwargs)

                if not isinstance(missing_data, pd.DataFrame):
                    raise ValueError(
                        "The decorated function must return a pandas DataFrame."
                    )

                if not missing_data.empty:
                    missing_data.index = pd.to_datetime(
                        missing_data.index
                    )  # Ensure the index is datetime

                # Update the cache
                cached_df = missing_data
                setattr(wrapper, f"cached_df_{cache_key}", cached_df)

                # Persist the updated cache to disk
                logger.debug(f"Saving cache to {cache_file}")
                cached_df.to_csv(cache_file)
            # Return the relevant slice of the cached DataFrame
            return_df = cached_df.loc[
                start_date : end_date.replace(hour=23, minute=59, second=59)
            ]
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
        raise ValueError("Please define either 'fixed_time' or 'fixed_nevents'")

    dates = cat.index.values
    if fixed_time is not None:
        ds = pd.Series(np.ones(len(dates)), index=dates)
        ds = pd.concat([ds, pd.Series([np.nan], index=[enddate])])
        ds.sort_index(inplace=True)
        ds = ds.rolling("{:d}D".format(fixed_time)).count() / fixed_time
        ds.index -= pd.Timedelta("{:d}D".format(int(fixed_time / 2.0)))
        return pd.DataFrame({"obs": ds}).tz_localize("utc")
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
        return pd.DataFrame({"obs": aRate}, index=aBin).tz_localize("utc")
    else:
        raise ValueError("Please define either 'fixed_time' or 'fixed_nevents'")


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


def hex_to_rgb(value, alpha=1.0):
    """Return (red, green, blue) for the color given as #rrggbb."""
    value = value.lstrip("#")
    lv = len(value)
    rgb_list = [int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)]
    rgb_list.append(alpha)
    return tuple(rgb_list)


def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return "#%02x%02x%02x" % (red, green, blue)


def get_color(idx, alpha=1.0, style="seaborn-v0_8-paper"):
    """Return a color from the matplotlib color cycle by index.

    Parameters
    ----------
    idx : int
        Index of the color to return.
    alpha : float, optional
        Opacity of the color between 0 and 1, by default 1.
    style : str, optional
        matplotlib style to choose colors from, by default 'bmh'

    Returns
    -------
    _type_
        _description_
    """
    matplotlib.style.use(style)
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors_rgb = []
    for c in colors:
        colors_rgb.append(hex_to_rgb(c, alpha=alpha))
    return f"rgba{colors_rgb[idx]}"


def test_signal(
    nsec=3600,
    sampling_rate=100.0,
    frequencies=[0.1, 3.0, 10.0],
    amplitudes=[0.1, 1.0, 0.7],
    phases=[0.0, np.pi * 0.25, np.pi],
    offsets=[0.0, 0.0, 0.0],
    starttime=UTCDateTime(1970, 1, 1),
    gaps=False,
    noise=True,
    noise_std=0.5,
    sinusoid=True,
    addchirp=True,
    network="NZ",
    station="BLUB",
    location="",
    channel="HHZ",
):
    """
    Produce a test signal for which we know where the peaks
    are in the spectrogram.

    :param nsec: Length of the trace in seconds.
    :type nsec: int
    :param sampling_rate: Sampling rate of the signal in Hz.
    :type sampling_rate: float
    :param starttime: Starttime of the trace.
    :type starttime: :class:`~obspy.UTCDateTime`
    :param gaps: If 'True' add gaps to the test signal.
    :type gaps: boolean
    :param noise_std: Standard deviation of the noise.
    :type noise_std: float
    :param sinusoid: Add a signal with a sinusoidal frequency change.
    :type sinusoid: bool
    :param station: Station name of the returned trace.
    :type station: str
    :param location: Location of the returned trace.
    :type location: str
    :return: Time series of test data
    :rtype: :class:`~obspy.Trace`
    """
    t = np.arange(0, nsec, 1 / sampling_rate)
    signals = []
    # Some constant signals with different phases
    for f, A, ph, dt in zip(frequencies, amplitudes, phases, offsets):
        _s = np.zeros(t.size)
        dt_idx = int(dt * sampling_rate)
        _s[dt_idx:] = A * np.sin(2.0 * np.pi * f * t[dt_idx:] + ph)
        signals.append(_s)
    if addchirp:
        # Frequency-swept signals
        # Chirp
        s4 = 0.8 * chirp(t, 5, t[-1], 15, method="quadratic")
        signals.append(s4)

    if sinusoid:
        vals = [6]
        for k in range(0, 7):
            vals.append(
                2
                * (-1) ** k
                * np.power(2 * np.pi / nsec, 2 * k + 1)
                / M.factorial(2 * k + 1)
            )
            vals.append(0)
        p = np.poly1d(np.array(vals[::-1]))
        s5 = sweep_poly(t, p)
        signals.append(s5)

    # add some noise
    if noise:
        rs = np.random.default_rng(42)
        noise = rs.normal(loc=0.0, scale=noise_std, size=t.size)
        for s in signals:
            noise += s
        signal = noise
    else:
        signal = signals[0]
        for s in signals[1:]:
            signal += s
    stats = {
        "network": network,
        "station": station,
        "location": location,
        "channel": channel,
        "npts": len(signal),
        "sampling_rate": sampling_rate,
        "mseed": {"dataquality": "D"},
    }
    stats["starttime"] = starttime
    stats["endtime"] = stats["starttime"] + nsec
    if gaps:
        p1 = (0.27, 0.33)
        p2 = (0.69, 0.83)
        p3 = (0.95, 1)
        for pmin, pmax in [p1, p2, p3]:
            idx0 = int(pmin * nsec * sampling_rate)
            idx1 = int(pmax * nsec * sampling_rate)
            signal[idx0:idx1] = np.nan
    return Trace(data=signal, header=stats)
