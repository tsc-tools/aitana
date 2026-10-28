"""
Compute the scattering transform of three-component seismic data. The
``ruapehu_snet`` workflow in aitana-pipelines fetches the data, runs it and
stores the coefficients.
"""
import logging
from obspy import UTCDateTime
from aitana.features.util import window_array
from scatseisnet import ScatteringNetwork
import numpy as np
from obspy import Stream, Trace
import xarray as xr
import pandas as pd
from aitana.features.feature_base import FeatureBaseClass

logger = logging.getLogger(__name__)

class SNet(FeatureBaseClass):
    def __init__(self, max_gap_fraction=0.1):
        super(SNet, self).__init__()
        self.segment_duration_seconds = 1200.0
        self.sampling_rate_hertz = 25.0
        self.overlap = 0.5
        # windows missing more data than this in any component are NaN
        self.max_gap_fraction = max_gap_fraction
        self.steplength = int(self.segment_duration_seconds*self.sampling_rate_hertz) - int(self.segment_duration_seconds*self.sampling_rate_hertz * self.overlap)
        self.samples_per_segment = int(self.segment_duration_seconds * self.sampling_rate_hertz)
        self.bank_keyword_arguments = (
            {"octaves": 6, "resolution": 4, "quality": 2},
            {"octaves": 6, "resolution": 1, "quality": 2},
        )
        msg = f"SNet initialized with segment_duration_seconds={self.segment_duration_seconds}, "
        msg += f"sampling_rate_hertz={self.sampling_rate_hertz}, overlap={self.overlap}, "
        msg += f"steplength={self.steplength}, samples_per_segment={self.samples_per_segment}" 
        logger.debug(msg)

    def preprocess(self, st: Stream, starttime: UTCDateTime, endtime: UTCDateTime) -> Stream:
        """
        Resample traces onto a common grid at ``sampling_rate_hertz``.

        Gaps (NaN) are filled with the last valid sample so that each trace can be
        detrended and low-pass filtered against aliasing, then the trace is
        interpolated onto the grid. Grid samples in a gap or outside the trace
        are NaN, for ``compute`` to mask.

        Parameters
        ----------
        st : obspy.Stream
            Traces of the components, e.g. Z, N, E.
        starttime : obspy.UTCDateTime
            First sample of the grid.
        endtime : obspy.UTCDateTime
            End of the grid (exclusive).

        Returns
        -------
        obspy.Stream
            One trace per input trace, each with
            ``(endtime - starttime) * sampling_rate_hertz`` samples from ``starttime``.
        """
        npts = int(round((endtime - starttime) * self.sampling_rate_hertz))
        out = Stream()
        for tr in st:
            tr = tr.copy()
            tr.data = tr.data.astype(float)
            missing = np.isnan(tr.data)
            data = np.full(npts, np.nan)
            if not missing.all():
                idx = np.where(~missing, np.arange(tr.stats.npts), np.argmin(missing))
                np.maximum.accumulate(idx, out=idx)
                tr.data = tr.data[idx]
                tr.detrend("linear")
                if tr.stats.sampling_rate > self.sampling_rate_hertz:
                    # zero-phase, 10 Hz at 25 Hz: about -5 dB at the highest filter
                    # centre (10.5 Hz), aliases at least 26 dB down
                    tr.filter("lowpass", freq=0.4 * self.sampling_rate_hertz, corners=4,
                              zerophase=True)
                grid = (starttime - tr.stats.starttime) + np.arange(npts) / self.sampling_rate_hertz
                data = np.interp(grid, tr.times(), tr.data, left=np.nan, right=np.nan)
                data[np.interp(grid, tr.times(), missing.astype(float)) > 0] = np.nan
            header = {k: tr.stats[k] for k in ("network", "station", "location", "channel")}
            out += Trace(data, header=dict(header, starttime=starttime,
                                           sampling_rate=self.sampling_rate_hertz))
        return out

    def compute(self, st: Stream):
        logger.debug("Computing scattering coefficients.")
        window_len = int(self.segment_duration_seconds * self.sampling_rate_hertz)
        noverlap = int(self.segment_duration_seconds * self.sampling_rate_hertz * self.overlap)
        results = []
        gap_fractions = []
        for tr in st[:3]:
            gap = np.isnan(tr.data)
            result, windowVals = window_array(np.where(gap, 0.0, tr.data), window_len, noverlap)
            results.append(result.T)
            gap_fractions.append(window_array(gap, window_len, noverlap, remove_mean=False,
                                              taper=False, padval=1).mean(axis=0))
        segments = np.stack(results, axis=1)
        timestamps = [(st[0].stats.starttime + _t).datetime for _t in np.arange(segments.shape[0])*self.steplength / self.sampling_rate_hertz]
        network = ScatteringNetwork(
            *self.bank_keyword_arguments,
            bins=self.samples_per_segment,
            sampling_rate=self.sampling_rate_hertz,
        )
        time_debug_start = UTCDateTime.now()
        scattering_coefficients = network.transform(segments, reduce_type=np.median)
        time_debug_end = UTCDateTime.now()
        logger.debug(f"Scattering coefficients computed in {time_debug_end - time_debug_start} seconds.")
        gappy = np.max(gap_fractions, axis=0) > self.max_gap_fraction
        for coefficients in scattering_coefficients:
            coefficients[gappy] = np.nan
        coords_order1 = {'datetime': pd.to_datetime(timestamps), 'channel': np.arange(scattering_coefficients[0].shape[1]),
                        'frequency': network.banks[0].centers}
        coords_order2 = coords_order1.copy()
        coords_order2['order2 frequency'] = network.banks[1].centers 
        xda_order1 = xr.DataArray(scattering_coefficients[0], dims=['datetime', 'channel', 'frequency'],
                                coords=coords_order1)
        xda_order2 = xr.DataArray(scattering_coefficients[1], dims=['datetime', 'channel', 'frequency', 'order2 frequency'],
                                coords=coords_order2)
        self.feature = xr.Dataset({'order1': xda_order1, 'order2': xda_order2})
        self.feature.attrs["starttime"] = st[0].stats.starttime.isoformat()
        self.feature.attrs["endtime"] = st[0].stats.endtime.isoformat()
        self.feature.attrs["station"] = st[0].stats.station
        return self.feature
