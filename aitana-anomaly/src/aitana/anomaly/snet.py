"""
Compute the scattering transform of three-component seismic data, embed its
coefficients in two dimensions and score how novel they are. The ``ruapehu_snet``
workflow in aitana-pipelines fetches the data, runs all three and stores the
coefficients, the embedding and the novelty index.
"""
import logging
from obspy import UTCDateTime
from aitana.features.util import window_array
from scatseisnet import ScatteringNetwork
import numpy as np
from obspy import Stream, Trace
import xarray as xr
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
import umap
from aitana.anomaly.anomaly_base import AnomalyDetectionBaseClass
from aitana.features.feature_base import FeatureBaseClass

logger = logging.getLogger(__name__)

# SNet windows span 20 min every 10 min: label scores and embedded windows by the
# window's centre, which places a sustained change close to its onset. A score
# can only be computed once its window has ended, 10 min after this label
LABEL_OFFSET = pd.Timedelta("10min")

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


def _features(snet):
    """
    Flatten the coefficients of each window into one feature vector.

    Parameters
    ----------
    snet : xr.Dataset
        ``order1`` and ``order2`` coefficients as ``SNet.compute`` returns them.

    Returns
    -------
    np.ndarray
        Shape (n_windows, n_features): the first-order, then the second-order
        coefficients of all channels.
    """
    order1 = snet["order1"].transpose("datetime", "channel", "frequency").values
    order2 = snet["order2"].transpose(
        "datetime", "channel", "frequency", "order2 frequency"
    ).values
    n = snet.sizes["datetime"]
    return np.hstack([order1.reshape(n, -1), order2.reshape(n, -1)])


def embed(snet, n_sources=10, random_state=42):
    """
    Embed windows of scattering coefficients in two dimensions.

    Following Steinmann et al. (2023, doi:10.1029/2023JB027167), the first- and
    second-order coefficients of each window are reduced to ``n_sources``
    independent components with FastICA, which UMAP then embeds in two
    dimensions. Both are fitted on ``snet`` itself, so a state unlike the rest
    can form an island of its own, which it can't when windows are projected
    onto an earlier fit. Windows with missing coefficients are left out.

    The coordinates belong to this fit: embedding again with more windows moves
    windows and their neighbours on the map, and can turn a map without islands,
    while what the map separates stays the same. Anything that sets windows
    apart shapes the map, so a change of instrument can split it as readily as
    a change of the volcano.

    Parameters
    ----------
    snet : xr.Dataset
        ``order1`` and ``order2`` coefficients, as ``SNet.compute`` returns them.
    n_sources : int
        Number of independent components.
    random_state : int
        Seed of FastICA and UMAP. A seeded UMAP fit runs single-threaded, about
        7 min for six years of windows.

    Returns
    -------
    xr.DataArray
        Embedding coordinates with dimensions (datetime, component), labelled by
        each window's label plus ``LABEL_OFFSET``, NaN for windows with missing
        coefficients.
    """
    X = _features(snet)
    finite = np.isfinite(X).all(axis=1)
    model = Pipeline([
        ("ica", FastICA(n_components=n_sources, random_state=random_state)),
        ("umap", umap.UMAP(random_state=random_state, n_jobs=1)),
    ])
    embedding = np.full((len(X), 2), np.nan)
    embedding[finite] = model.fit_transform(X[finite])
    return xr.DataArray(
        embedding,
        dims=["datetime", "component"],
        coords={
            "datetime": pd.DatetimeIndex(snet["datetime"].values) + LABEL_OFFSET,
            "component": [0, 1],
        },
    )


class SNetNovelty(AnomalyDetectionBaseClass):
    """
    Score how unlike a reference period each window of scattering coefficients is.

    The coefficients of each window are reduced to ``n_sources`` independent
    components with FastICA fitted on the reference windows, as in
    ``embed``. The raw score of a window is its Euclidean distance in
    that space to the nearest reference window more than ``exclusion`` away in
    time. FastICA only rotates the whitened principal components, so these are
    the distances between whitened principal components. The exclusion leaves out a reference window's overlapping neighbours,
    so the reference windows' own scores, against which raw scores are ranked
    to ``[0, 1]`` with ``normalize_score()``, are comparable to those of new
    windows.

    The reference is fixed, so a new state stays novel for as long as it lasts,
    unlike a trailing history that adapts. That includes changes of the station
    rather than the volcano: after FWVZ's sensor and datalogger swap on
    2025-02-03 the horizontal coefficients roughly doubled, and a 2021
    reference scores much of the later data as novel, which is why
    ``ruapehu_snet`` scores Z alone for now. A state seen in only one
    reference window already counts as familiar. Each window is scored on its
    own, so short transients such as regional earthquakes or storms can reach
    high scores too.

    Parameters
    ----------
    n_sources : int
        Number of independent components.
    exclusion : str or pd.Timedelta
        Reference windows this close in time to a scored window are ignored.
    random_state : int
        Seed of FastICA.

    Attributes
    ----------
    distance_ : pd.Series
        Raw score from the last ``compute_anomaly_index`` call, on the same
        index and NaN in the same places as the anomaly index.
    """

    def __init__(self, n_sources=10, exclusion="10min", random_state=42):
        super().__init__(features=None)
        self.n_sources = n_sources
        self.exclusion = pd.Timedelta(exclusion)
        self.random_state = random_state

    def fit(self, snet):
        """
        Fit the independent components and the reference scores.

        Unlike the base class, this takes the reference windows' coefficients
        rather than a time span.

        Parameters
        ----------
        snet : xr.Dataset
            ``order1`` and ``order2`` coefficients of the reference windows, as
            ``SNet.compute`` returns them. Windows with missing coefficients
            are left out.

        Returns
        -------
        SNetNovelty
            The fitted detector.
        """
        X = _features(snet)
        finite = np.isfinite(X).all(axis=1)
        self.ica_ = FastICA(n_components=self.n_sources, random_state=self.random_state)
        sources = self.ica_.fit_transform(X[finite])
        self.times_ = snet["datetime"].values[finite].astype("datetime64[ns]").astype(np.int64)
        self.nn_ = NearestNeighbors().fit(sources)
        reference = self._distance(sources, self.times_)
        self.reference_ = reference[np.isfinite(reference)]
        return self

    def _distance(self, sources, times):
        """
        Distance to the nearest reference window more than ``exclusion`` away.

        Parameters
        ----------
        sources : np.ndarray
            Independent components of the windows, (n_windows, n_sources).
        times : np.ndarray
            Labels of the windows in ns since the epoch.

        Returns
        -------
        np.ndarray
            One distance per window, NaN if every reference window is too close.
        """
        out = np.full(len(sources), np.nan)
        todo = np.arange(len(sources))
        k = 16
        while len(todo):
            k = min(k, len(self.times_))
            dist, idx = self.nn_.kneighbors(sources[todo], n_neighbors=k)
            far = np.abs(self.times_[idx] - times[todo, None]) > self.exclusion.value
            found = far.any(axis=1)
            out[todo[found]] = dist[found, far[found].argmax(axis=1)]
            todo = todo[~found]
            if k == len(self.times_):
                break
            k *= 4
        return out

    def compute_anomaly_index(self, snet):
        """
        Compute the novelty index of windows of scattering coefficients.

        Parameters
        ----------
        snet : xr.Dataset
            ``order1`` and ``order2`` coefficients, as ``SNet.compute`` returns
            them, each window labelled by its first sample.

        Returns
        -------
        pd.Series
            Novelty index in ``[0, 1]`` labelled by each window's label plus
            ``LABEL_OFFSET``, NaN for windows with missing coefficients.
        """
        X = _features(snet)
        finite = np.isfinite(X).all(axis=1)
        times = snet["datetime"].values.astype("datetime64[ns]").astype(np.int64)
        distance = np.full(len(X), np.nan)
        if finite.any():
            distance[finite] = self._distance(self.ica_.transform(X[finite]), times[finite])
        aindex = self.normalize_score(distance, self.reference_)
        aindex[~np.isfinite(distance)] = np.nan
        labels = pd.DatetimeIndex(snet["datetime"].values) + LABEL_OFFSET
        self.distance_ = pd.Series(distance, index=labels, name="distance")
        return pd.Series(aindex, index=labels, name="anomaly_index")
