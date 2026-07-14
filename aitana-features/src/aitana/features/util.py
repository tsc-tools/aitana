"""
Functionality common to several modules.
"""

import logging
import re

import numpy as np
import pandas as pd
from obspy import UTCDateTime

logger = logging.getLogger(__name__)


def stride_windows(x, n, noverlap):
    """
    Get all windows of x with length n as a single array,
    using strides to avoid data duplication.

    :param x: 1D array or sequence containing the data.
    :type x: :class:`~numpy.ndarray`
    :param n: The number of data points in each window.
    :type n: int
    :param noverlap: The overlap between adjacent windows.
                     Default is 0 (no overlap)
    :type noverlap: int
    :return: Array with the windowed data in columns.
    :rtype: :class:`~numpy.ndarray`
    """

    if noverlap >= n:
        raise ValueError("noverlap must be less than n")
    if n < 1:
        raise ValueError("n cannot be less than 1")

    if n == 1 and noverlap == 0:
        return x[np.newaxis]
    if n > x.size:
        raise ValueError("n cannot be greater than the length of x")

    # np.lib.stride_tricks.as_strided easily leads to memory corruption for
    # non integer shape and strides, i.e. noverlap or n. See #3845.
    noverlap = int(noverlap)
    n = int(n)

    step = n - noverlap
    shape = (n, (x.shape[-1] - noverlap) // step)
    strides = (x.strides[0], step * x.strides[0])
    return np.lib.stride_tricks.as_strided(
        x, shape=shape, strides=strides, writeable=False
    )


def demean(x, axis=None):
    """
    Return x minus the mean(x).

    :param x: array or sequence containing the data can
              have any dimensionality
    :type x: :class:`~numpy.ndarray`
    :param axis: The axis along which to take the mean. See numpy.mean
                 for a description of this argument.
    :type axis: int
    :return: input minus the arithmetic mean along the specified axis.
    :rtype: :class:`~numpy.ndarray`
    """
    x = np.asarray(x)

    if axis is not None and axis + 1 > x.ndim:
        raise ValueError("axis(=%s) out of bounds" % axis)

    return x - np.nanmean(x, axis=axis, keepdims=True)


def apply_hanning(x, return_window=None):
    """
    Apply a hanning window to the given 1D or 2D array along
    the given axis.

    :param x: 1-D or 2-D array or sequence containing the data
    :type x: :class:`numpy.ndarray`
    :param return_window: If true, also return the 1D values of the window
                          that was applied.
    :type return_window: bool
    :return: **wtr[, wvals]** input multiplied with the hanning
             window function, hanning window function
    """
    x = np.asarray(x)

    if x.ndim < 1 or x.ndim > 2:
        raise ValueError("only 1D or 2D arrays can be used")

    xshapetarg = x.shape[0]
    windowVals = np.hanning(xshapetarg) * np.ones(xshapetarg, dtype=x.dtype)

    if x.ndim == 1:
        if return_window:
            return windowVals * x, windowVals
        else:
            return windowVals * x

    windowVals = windowVals[:, np.newaxis]

    if return_window:
        return windowVals * x, windowVals
    else:
        return windowVals * x


def window_array(
    x, nwin, noverlap, remove_mean=True, taper=True, padval=0, return_window=True
):
    """
    Take a 1-D numpy array and devide it into (overlapping)
    windows.

    :param x: 1D array containing the data.
    :type x: :class:`~numpy.ndarray`
    :param n: The number of data points in each window. If this
              is not a power of 2, the length of each window will be
              the nearest power of 2.
    :type n: int
    :param noverlap: The number of overlapping points between
                     adjacent windows.
    :type noverlap: int
    :param remove_mean: Remove the individual mean from each
                        window.
    :type remove_mean: bool
    :param taper: Apply a hanning taper to each window.
    :type taper: bool
    :param return_window: If taper is 'True', return the values
                          of the hanning taper.
    :type return_window: bool
    :param padval: If this is not 'None', padval will be used to
                   pad the input array such that its length is
                   a multiple of the window length.
    :type padval: float, class:`numpy.nan`, or None
    :return: Array with the windowed data in columns.
    :rtype: :class:`~numpy.ndarray`
    """
    x = x.astype("float")
    npts = x.size
    if nwin > npts:
        msg = "window length can't be greater than array length"
        raise ValueError(msg)

    # step length that the window is advanced by
    step = nwin - noverlap
    # compute the remainder of the trace after dividing it
    # into windows
    rest = (npts - nwin) % step

    # if the remainder is >0 pad with zeros
    if rest > 0:
        npts_pad = npts + (step - rest)
        x = np.resize(x, int(npts_pad))
        x[npts:] = padval

    result = stride_windows(x, nwin, noverlap)
    if remove_mean:
        result = demean(result, axis=0)
    if taper:
        result, windowVals = apply_hanning(result, return_window=True)
        if return_window:
            return result, windowVals
    return result


def trace_window_times(trace, window_len, min_len=0.8):
    """Subdivide an `obspy.core.trace.Trace` into windows of `window_len`
    length and iterate over window start & end times.

    :param trace: The data to be windowed.
    :type trace: :class:`obspy.core.trace.Trace`
    :param window_len: The window length in seconds.
    :type window_len: float
    :param min_len: Windows with length < `min_len * window_len` are
        skipped, defaults to 0.8
    :type min_len: float, optional
    :yield: The window start & end times
    :rtype: :class:`obspy.core.utcdatetime.UTCDateTime`,
        :class:`obspy.core.utcdatetime.UTCDateTime`,
    """

    min_len = float(min_len)
    if not 0 < min_len <= 1.0:
        raise ValueError("min_len <= 0 or > 1.")
    min_len = min_len * window_len

    if (trace.stats.endtime - trace.stats.starttime) < min_len:
        raise ValueError("window length > total trace data.")

    t0 = trace.stats.starttime
    while t0 < trace.stats.endtime:
        t1 = min(t0 + window_len, trace.stats.endtime)
        this_window_len = t1 - t0
        if this_window_len >= min_len:
            if this_window_len < window_len:
                t0 = t1 - window_len
            yield t0, t1

        t0 += window_len


def trace_window_data(trace, window_len, min_len=0.8):
    """Subdivide an `obspy.core.trace.Trace` into windows of `interval`
    length and iterate over windows.

    :param trace: The data to be windowed.
    :type trace: :class:`obspy.core.trace.Trace`
    :param window_len: The window length in seconds.
    :type window_len: float
    :param min_len: Windows with length < `min_length * window_len` are
        skipped, defaults to 0.8
    :type min_len: float, optional
    :yield: A trace data windowNone
    :rtype: `obspy.core.trace.Trace`
    """

    for t0, t1 in trace_window_times(trace, window_len, min_len):
        yield trace.slice(t0, t1, nearest_sample=False)


def apply_freq_filter(
    data, filtertype, filterfreq, corners=4, zerophase=False, **options
):
    """Apply frequency filter to seismic data.

    Filter parameters are taken from instance attributes
    `filtertype` and `filterfreq`.

    :param data: The seismic data to be filtered.
    :type data: :class:`obspy.core.trace.Trace` or
        :class:``obspy.core.stream.Stream`
    :param corners: Filter corners / order, defaults to 4.
    :type corners: int
    :param zerophase: If True, apply filter once forwards and once
        backwards. Defaults to False.
    :type zerophase: bool
    :param options: Passed to underlying obspy filter methods.
    """
    f_min, f_max = filterfreq
    options.update({"zerophase": zerophase, "corners": corners})

    if filtertype in ["lp", "lowpass"]:
        if f_max is None:
            raise ValueError("Define upper frequency cutoff for low-pass filter.")
        data.filter("lowpass", freq=f_max, **options)

    elif filtertype in ["hp", "highpass"]:
        if f_min is None:
            raise ValueError("Define lower frequency cutoff for high-pass filter.")
        data.filter("highpass", freq=f_min, **options)

    elif filtertype in ["bp", "bandpass"]:
        if f_min is None or f_max is None:
            raise ValueError(
                "Define upper and lower requency cutoff for band-pass filter."
            )
        data.filter("bandpass", freqmin=f_min, freqmax=f_max, **options)

    elif filtertype is not None:
        raise ValueError(f"Unrecognised filtertype '{filtertype}'")


def round_time(time, interval):
    """
    Find closest multiple of interval to time.

    :param time: Time to round
    :type time: :class:`obspy.UTCDateTime`
    :param interval: Interval length in seconds
    :type interval: int
    :returns: The new time that is a multiple of interval
    :rtype: :class:`obspy.UtCDateTime`
    """
    tmp = (time + 0.5 * interval).timestamp % interval
    ntime = (time + 0.5 * interval).timestamp - tmp
    return UTCDateTime(ntime)


def stack_feature(xrdata, stack_length, interval="10min"):
    """
    Stack features in a dataarray.

    Parameters
    ----------
    xrdata : xarray.DataArray
        The data array to stack.
    stack_length : str
        The length of the stack.
    interval : str, optional
        The interval between stacks, by default "10min".
    """
    num_periods = None
    valid_stack_units = ["W", "D", "h", "T", "min", "S"]
    if re.match(r"\d*\s*(\w*)", stack_length).group(1) not in valid_stack_units:
        raise ValueError(
            "Stack length should be one of: {}".format(", ".join(valid_stack_units))
        )

    if pd.to_timedelta(stack_length) < pd.to_timedelta(interval):
        raise ValueError(
            "Stack length {} is less than interval {}".format(stack_length, interval)
        )

    num_periods = pd.to_timedelta(stack_length) / pd.to_timedelta(interval)
    if not num_periods.is_integer():
        raise ValueError(
            "Stack length {} / interval {} = {}, but it needs"
            " to be a whole number".format(stack_length, interval, num_periods)
        )

    # Stack features
    logger.debug("Stacking feature...")
    try:
        xdf = xrdata.rolling(
            datetime=int(num_periods), center=False, min_periods=1
        ).mean()
    except ValueError as e:
        logger.error(e)
        logger.error(
            "Stack length {} is not valid for feature {}".format(
                stack_length, xrdata.name
            )
        )
    else:
        return xdf
