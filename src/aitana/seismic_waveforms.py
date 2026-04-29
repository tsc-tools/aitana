from datetime import date, datetime, timedelta, timezone
import inspect
import io
import logging
import os
from pathlib import Path
import warnings

from torch.utils.data import Dataset
import torch
import fsspec
import numpy as np
from obspy import Inventory, Stream, Trace, UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.filesystem.sds import Client as SDS_Client

from aitana.util import test_signal


logger = logging.getLogger(__name__)


class PostProcessException(Exception):
    pass


class PostProcess:
    """
    Class for checking waveforms and station metadata for valid values,
    filling waveform data gaps and removing instrument sensitivity
    """

    def __init__(
        self,
        st=Stream(),
        inv=Inventory(),
        inv_dt=Inventory(),
        startdate=None,
        enddate=None,
        loc="*",
        comp="*Z",
        fill_value=np.nan,
    ):
        self.st = st
        self.inv = inv
        self.inv_dt = inv_dt
        self.startdate = startdate
        self.enddate = enddate
        self.loc = loc
        self.comp = comp
        self.fill_value = fill_value

        # Get check registry
        self.checklist = [
            x
            for x in inspect.getmembers(self)
            if inspect.ismethod(x[1]) and x[0].startswith("check")
        ]
        # Get matching post-processing functions
        self.pp_functions = []
        for check in self.checklist:
            self.pp_functions.append(
                [
                    x
                    for x in inspect.getmembers(self)
                    if x[0] == check[0].replace("check", "output")
                ][0]
            )

        # Run checks
        self.res = None

    def run_post_processing(self):
        self.res = self.run_checks()
        func = [a[1] for (a, b) in zip(self.pp_functions, self.res) if b][0]
        trace = func()
        return trace

    def run_checks(self):
        self.res = [check[1]() for check in self.checklist]
        if sum(self.res) == 0:
            raise NotImplementedError(
                "No post processing not set up for this case:\n" "{}".format(
                    self.st)
            )
        elif sum(self.res) > 1:
            msg = (
                "More than one check is true: {}. Processing workflow "
                "is unclear.".format(
                    ", ".join([a[0] for (a, b) in zip(
                        self.checklist, self.res) if b])
                )
            )
            raise PostProcessException(msg)
        else:
            return self.res

    def check_case_no_data(self):
        """
        Checks if either no stream, or no station metadata exist.
        """
        if len(self.st) < 1 or len(self.inv_dt) < 1:
            return True
        elif (
            len(self.inv_dt[0][0]) == 1
            and self.inv_dt[0][0][0].start_date == self.enddate
        ):
            return True
        elif (
            len(self.inv_dt[0][0]) == 1
            and self.inv_dt[0][0][0].end_date == self.startdate
        ):
            return True
        else:
            return False

    def output_case_no_data(self):
        raise PostProcessException

    def check_case_multiple_channel_periods(self):
        """
        Checks if there are multiple channel operation periods in a given
        time period. If so, different response removal is required.
        """
        if len(self.inv_dt) < 1 or len(self.st) < 1:
            return False
        elif len(self.inv_dt[0][0]) > 1:
            return True
        else:
            return False

    def output_case_multiple_channel_periods(self):
        """
        Stitches two response periods together to make a single merged
        trace.
        """
        if len(self.inv_dt[0][0]) > 2:
            raise NotImplementedError(
                "Current post processing function only valid for a "
                "maximum of two response periods."
            )
        st1 = self.st.copy()
        st1.merge(fill_value=self.fill_value)
        st2 = st1.copy()
        st1.trim(
            starttime=self.startdate,
            endtime=self.inv_dt[0][0][0].end_date,
            nearest_sample=False,
        )
        st2.trim(
            starttime=self.inv_dt[0][0][1].start_date,
            endtime=self.enddate,
            nearest_sample=False,
        )
        st1.attach_response(self.inv_dt)
        st1.remove_sensitivity()
        st2.attach_response(self.inv_dt)
        st2.remove_sensitivity()
        st1 += st2
        st1.merge(fill_value=self.fill_value)
        st1.trim(
            self.startdate,
            self.enddate,
            nearest_sample=False,
            pad=True,
            fill_value=self.fill_value,
        )
        return st1[0]

    def check_case_incomplete_station_metadata(self):
        """
        The case where a a channel period start/stops during the
        selected time window. Some parts of the trace in the stream
        might not correspond to a valid period according to station
        metadata.
        """
        if len(self.inv_dt) < 1:
            return False
        elif len(self.inv_dt[0][0]) != 1 or len(self.st) < 1:
            return False
        elif (self.startdate < self.inv_dt[0][0][0].start_date < self.enddate) or (
            self.startdate < self.inv_dt[0][0][0].end_date < self.enddate
        ):
            return True
        else:
            return False

    def output_case_incomplete_station_metadata(self):
        if self.inv_dt[0][0][0].start_date > self.startdate:
            self.st.trim(
                self.inv_dt[0][0][0].start_date, self.enddate, nearest_sample=False
            )
            self.st.attach_response(self.inv_dt)
            self.st.remove_sensitivity()
            self.st.trim(
                self.startdate,
                self.enddate,
                nearest_sample=False,
                pad=True,
                fill_value=self.fill_value,
            )
            return self.st[0]
        elif self.inv_dt[0][0][0].end_date < self.enddate:
            self.st.trim(
                self.startdate, self.inv_dt[0][0][0].end_date, nearest_sample=False
            )
            self.st.attach_response(self.inv_dt)
            self.st.remove_sensitivity()
            self.st.trim(
                self.startdate,
                self.enddate,
                nearest_sample=False,
                pad=True,
                fill_value=self.fill_value,
            )
            return self.st[0]
        else:
            raise PostProcessException(
                "Post-procesing function not equipped to" "handle:\n" "{}".format(
                    self.st
                )
            )

    def check_case_no_station_issues(self):
        if len(self.inv_dt) < 1:
            return False
        elif len(self.inv_dt[0][0]) != 1 or len(self.st) < 1:
            return False
        elif (
            len(self.inv_dt[0][0]) == 1
            and self.inv_dt[0][0][0].end_date == self.startdate
        ):
            return False
        elif (
            len(self.inv_dt[0][0]) == 1
            and self.inv_dt[0][0][0].start_date == self.enddate
        ):
            return False
        elif (self.startdate < self.inv_dt[0][0][0].start_date < self.enddate) or (
            self.startdate < self.inv_dt[0][0][0].end_date < self.enddate
        ):
            return False
        else:
            return True

    def output_case_no_station_issues(self):
        self.st.merge(fill_value=self.fill_value)
        self.st.trim(
            starttime=self.startdate,
            endtime=self.enddate,
            nearest_sample=False,
            pad=True,
            fill_value=self.fill_value,
        )
        self.st.attach_response(self.inv_dt)
        self.st.remove_sensitivity()
        return self.st[0]


def get_instrument_response(
    net,
    site,
    loc,
    staxml_dir="./instrument_response",
    fdsn_urls=("https://service.geonet.org.nz"),
):
    """
    Retrieve instrument response file in STATIONXML format.
    """
    fn = "{:s}.xml".format(site)
    fn = os.path.join(staxml_dir, fn)
    try:
        inv = read_inventory(fn, format="STATIONXML")
    except FileNotFoundError:
        try:
            os.makedirs(staxml_dir)
        except FileExistsError:
            pass
        for url in fdsn_urls:
            try:
                client = FDSN_Client(base_url=url)
                inv = client.get_stations(
                    network=net,
                    location=loc,
                    station=site,
                    channel="*",
                    level="response",
                )
            except Exception as e:
                logger.info(e)
            else:
                break
        inv.write(fn, format="STATIONXML")
    return inv


class WaveformBaseclass:
    pass


class SDSWaveforms(WaveformBaseclass):
    def __init__(self, sds_dir, fdsn_urls, staxml_dir, fill_value=np.nan):
        """
        Get seismic waveforms from a local SDS archive.
        """
        self.client = SDS_Client(sds_dir)
        self.fdsn_urls = fdsn_urls
        self.staxml_dir = staxml_dir
        self.fill_value = fill_value

    def get_waveforms(self, net, site, loc, comp, startdate, enddate):
        """
        :param fill_value: Default value used to fill gaps and pad traces to
                           cover the whole time period defined by `startdate`
                           and `enddate`. If 'interpolate' is chosen, the
                           traces will not be padded.
        :type fill_value: int, float, 'interpolate', np.nan, or None
        """
        inv = get_instrument_response(
            net, site, loc, staxml_dir=self.staxml_dir, fdsn_urls=self.fdsn_urls
        )

        st = self.client.get_waveforms(
            net, site, loc, comp, startdate, enddate, dtype="float64"
        )

        inv_dt = inv.select(
            location=loc, channel=comp, starttime=startdate, endtime=enddate - 1
        )

        # Post-process to remove sensitivity and account for gaps
        pp = PostProcess(
            st=st,
            inv=inv,
            inv_dt=inv_dt,
            startdate=startdate,
            enddate=enddate,
            loc=loc,
            comp=comp,
            fill_value=self.fill_value,
        )
        tr = pp.run_post_processing()
        return tr


class MockSDSWaveforms(WaveformBaseclass):
    """
    Mock SDSWaveforms class for testing by creating
    synthetic data for the requested streams.
    """

    def __init__(self, sds_dir):
        """
        Get seismic waveforms from a local SDS archive.
        """
        os.makedirs(sds_dir, exist_ok=True)
        self.client = SDS_Client(sds_dir)

    def save2sds(self, trace: Trace, rootdir: str):
        """
        Save a trace to a SDS directory structure.

        Parameters
        ----------
        trace : `obspy.Trace`
            Seismic/acoustic trace to be written to disk.
        rootdir : str
            Root directory for the SDS directory structure.
        """
        sds_fmtstr = os.path.join(
            "{year}",
            "{network}",
            "{station}",
            "{channel}.{sds_type}",
            "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}",
        )

        fullpath = sds_fmtstr.format(
            year=trace.stats.starttime.year,
            doy=trace.stats.starttime.julday,
            sds_type="D",
            **trace.stats,
        )
        fullpath = os.path.join(rootdir, fullpath)
        dirname, filename = os.path.split(fullpath)
        os.makedirs(dirname, exist_ok=True)
        print("writing", fullpath)
        trace.write(fullpath, format="mseed")

    def generate_dataset(
        self,
        rootdir: str,
        net: str,
        site: str,
        loc: str,
        comp: str,
        start: str,
        end: str,
    ) -> str:
        """
        Generate a test dataset for the integration tests.

        Parameters
        ----------
        rootdir : str
            Parent directory for the test dataset.
        start : str
            Start time for the test dataset in ISO 8601 format.
        end : str
            End time for the test dataset in ISO 8601 format.
        """
        tstart = UTCDateTime(start)
        # Always generate whole day files
        _tstart = UTCDateTime(year=tstart.year, julday=tstart.julday)
        tend = UTCDateTime(end)
        tr = test_signal(
            starttime=_tstart,
            sampling_rate=10,
            nsec=86400,
            gaps=True,
            network=net,
            station=site,
            location=loc,
            channel=comp,
        )
        while _tstart < tend:
            tr.stats.starttime = _tstart
            self.save2sds(tr, rootdir)
            _tstart += 86400

    def get_waveforms(self, net, site, loc, comp, startdate, enddate):
        """
        Generate synthetic daily files that start at midnight on the
        startdate and end at midnight on the enddate. The return the requested
        timespan from these files.
        """
        self.generate_dataset(
            self.client.sds_root, net, site, loc, comp, startdate, enddate
        )
        st = self.client.get_waveforms(
            net, site, loc, comp, startdate, enddate, dtype="float64"
        )
        return st[0]


class FDSNWaveforms(WaveformBaseclass):
    def __init__(self, url, debug=False, fill_value=np.nan):
        """
        Get seismic waveforms from FDSN web service.
        """
        self.client = FDSN_Client(base_url=url, debug=debug)
        self.fill_value = fill_value

    def get_waveforms(self, net, site, loc, comp, startdate, enddate):
        st = self.client.get_waveforms(
            net, site, loc, comp, startdate, enddate, attach_response=True
        )
        # Prepare seismic time series
        st.remove_sensitivity()
        # in case stream has more than one trace
        st.merge(fill_value=self.fill_value)
        if self.fill_value != "interpolate":
            st.trim(
                startdate,
                enddate,
                pad=True,
                fill_value=self.fill_value,
                nearest_sample=False,
            )
        return st[0]


class S3Waveforms(WaveformBaseclass):
    def __init__(self, staxml_dir, debug=False, fill_value=np.nan):
        """
        Get seismic waveforms from GeoNet's open data archive on AWS S3.

        Parameters
        ----------
        staxml_dir : str
            Directory where the station metadata in STATIONXML format is stored.
        debug : bool
            If True, debug messages will be printed to the console.
        fill_value : int, float, 'interpolate', np.nan, or None
            Value to use for filling gaps in the data.
        """
        self.fsclient = fsspec.filesystem("filecache", target_protocol='s3', target_options={'anon': True},
                                          cache_storage='/tmp/s3filescache')
        self.fdsn_urls = ['https://service.geonet.org.nz',
                          'https://service-nrt.geonet.org.nz']
        self.staxml_dir = staxml_dir
        self.debug = debug
        self.fill_value = fill_value
        self.bucket = "geonet-open-data"

    def get_waveforms(self, net, site, loc, comp, start, end):
        """
        Download waveforms.

        Parameters
        ----------
        net : str
            Network code.
        site : str
            Station code.
        loc : str
            Location code.
        comp : str
            Channel code.
        start : UTCDateTime
            Start time of the data to be downloaded.
        end : UTCDateTime
            End time of the data to be downloaded.

        Returns
        -------
        Trace
            An ObsPy Trace object containing the waveform data with sensitivity
            removed.
        """
        PATH_FORMAT = "{bucket}/waveforms/miniseed/{year}/{year}.{julday:03d}/{station}."
        PATH_FORMAT += "{network}/{year}.{julday:03d}.{station}."
        PATH_FORMAT += "{location}-{channel}.{network}.D"
        t_start = start
        t_end = UTCDateTime(
            year=t_start.year,
            julday=t_start.julday + 1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        t_end = min(t_end, end)
        st = Stream()
        while t_start < end:
            s3_file_path = PATH_FORMAT.format(
                bucket=self.bucket,
                year=t_start.year,
                julday=t_start.julday,
                station=site,
                network=net,
                location=loc,
                channel=comp,
            )
            if self.debug:
                logger.debug("Requesting {}".format(s3_file_path))
            with self.fsclient.open(s3_file_path) as f:
                _st = read(f, dtype="float64")
            _st.trim(t_start, t_end, nearest_sample=False)
            st += _st
            t_start = t_end
            t_end = t_start + timedelta(days=1)
            t_end = min(t_end, end)

        st.merge(fill_value=self.fill_value)
        inv = get_instrument_response(
            net, site, loc, staxml_dir=self.staxml_dir, fdsn_urls=self.fdsn_urls
        )
        inv_dt = inv.select(
            location=loc, channel=comp, starttime=t_start, endtime=t_end
        )

        # Post-process to remove sensitivity and account for gaps
        pp = PostProcess(
            st=st,
            inv=inv,
            inv_dt=inv_dt,
            startdate=start,
            enddate=end,
            loc=loc,
            comp=comp,
            fill_value=self.fill_value,
        )
        tr = pp.run_post_processing()
        return tr


class SeismicWaveforms(Dataset):
    def __init__(self, stream_id: str, clients: list[WaveformBaseclass],
                 start: UTCDateTime, end: UTCDateTime,
                 window_size: int, stride: int, cache_dir=None):

        self.net, self.site, self.loc, self.comp = stream_id.split(".")
        self.window_size = window_size
        self.stride = stride
        self.start = UTCDateTime(start)
        self.end = UTCDateTime(end)

        if len(clients) < 1:
            msg = "At least one data client is required."
            raise ValueError(msg)

        for client in clients:
            if not issubclass(client.__class__, WaveformBaseclass):
                msg = "All clients must be derived from WaveformBaseclass."
                raise ValueError(msg)

        self.clients = clients

        self.cache_dir = os.path.join(Path.home(), ".aitana_cache")
        if cache_dir is not None:
            self.cache_dir = cache_dir

        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_client = SDS_Client(self.cache_dir)

    def __len__(self):
        total_seconds = int((self.end - self.start))
        rest = (total_seconds - self.window_size) % self.stride
        return (total_seconds - self.window_size + self.stride - rest) // self.stride + 1

    def __getitem__(self, index):
        window_start = self.start + timedelta(seconds=index * self.stride)
        window_end = window_start + timedelta(seconds=self.window_size)
        if window_end > self.end:
            raise IndexError("Index out of range")
        # Always fetch whole days so we don't cache multiples files
        # for one day
        start_date = UTCDateTime(window_start.date)
        end_date = UTCDateTime(window_end.date) + 86400.0
        st = self.get_waveforms(self.net, self.site,
                                self.loc, self.comp, start_date, end_date)
        st.trim(
            starttime=window_start,
            endtime=window_end,
            nearest_sample=True,
            pad=True,
            fill_value=np.nan,
        )
        length = self.window_size * st[0].stats.sampling_rate
        tstart = st[0].stats.starttime.timestamp
        times = torch.linspace(tstart, tstart + st[0].stats.delta * self.window_size,
                               steps=int(length))
        traces = np.array([tr.data[:int(length)] for tr in st])
        return torch.vstack((torch.from_numpy(traces), times))

    def to_sds(self, tr: Trace):
        """
        Save trace to SDS directory.
        """
        start = tr.stats.starttime
        end = tr.stats.endtime
        sds_fmtstr = os.path.join(
            "{year}",
            "{network}",
            "{station}",
            "{channel}.{sds_type}",
            "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}",
        )

        current_date = start.date
        while current_date <= end.date:
            _tr = tr.slice(UTCDateTime(current_date),
                           UTCDateTime(current_date) + 86400)
            # print(_tr.stats.starttime, _tr.stats.endtime)
            fullpath = sds_fmtstr.format(
                year=_tr.stats.starttime.year,
                doy=_tr.stats.starttime.julday,
                sds_type="D",
                **_tr.stats,
            )
            fullpath = os.path.join(self.cache_dir, fullpath)
            dirname, _ = os.path.split(fullpath)
            os.makedirs(dirname, exist_ok=True)
            logger.debug(f"writing {_tr} to {fullpath}")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                _tr.write(fullpath, format="MSEED")
            current_date += timedelta(days=1)

    def get_waveforms(self, net, site, loc, comp, start, end, cache=False):
        """Yield waveform data for requested time span.

        Parameters
        ----------
        net, site, loc, comp : str
            Network, station, location and channel codes. ``comp`` may include a
            simple wildcard of the form ``??`` or ``?`` or patterns like ``HH?``
            to request multiple components (e.g. ``HH?`` -> ``HHZ, HHN, HHE``).
            Only a single wildcard character is currently supported and will
            expand to the common three-component set ``Z, N, E`` (or ``Z, 1, 2``
            if those exist in the data – both tried).
        start, end : UTCDateTime
            Start and end times.
        cache : bool
            If True, fetched data are written to the local SDS-style cache.

        Yields
        ------
        Trace or Stream
            For a single component request, yields an ObsPy ``Trace`` (backwards
            compatible). For a wildcard multi-component request, yields an
            ObsPy ``Stream`` with one Trace per component for each chunk.
        """
        # Detect simple wildcard request
        wildcard_request = ('?' in comp) or ('*' in comp)
        component_codes = [comp]
        if wildcard_request:
            # Only support patterns where wildcard replaces exactly one character
            # Typical use-case: 'HH?' or 'BH?'. Expand into Z,N,E first; if no
            # data found for N/E try 1/2 as horizontal components.
            base = comp.replace('*', '?')  # treat * like ? for single-char
            if base.count('?') == 1:
                pre, _ = base.split('?')
                candidate_sets = [['Z'], ['N', 'E'], ['1', '2']]
                expanded = []
                for candidates in candidate_sets:
                    expanded += [f"{pre}{c}" for c in candidates]
                    # Use first candidate set that returns at least one trace later
                    # We'll attempt all; selection logic implemented below
                component_codes = expanded
            else:
                # Fallback: if multiple wildcards, revert to original behavior
                wildcard_request = False
                component_codes = [comp]

        def _fetch_single_component(single_comp, t_start, t_end):
            """Inner helper replicating previous single-component logic.
            Returns a Trace (may be empty Trace() if not found)."""
            tr = Trace()
            try:
                if not cache:
                    raise AttributeError
                st = self.cache_client.get_waveforms(
                    net, site, loc, single_comp, t_start, t_end, dtype="float64"
                )
                tr = st.merge(fill_value=np.nan)[0]
                t_diff = int(t_end - t_start)
                if abs(t_diff - (tr.stats.npts - 1) * tr.stats.delta) > 1:
                    raise IndexError
                tr.stats["cached"] = True
            except (IndexError, AttributeError):
                msg = "Data for {} between {} and {} not found in cache."
                logger.debug(msg.format(
                    ".".join((net, site, loc, single_comp)), t_start, t_end))
                for client in self.clients:
                    try:
                        tr = client.get_waveforms(
                            net, site, loc, single_comp, t_start, t_end)
                    except Exception as e:
                        logger.info(e)
                        continue
                    else:
                        break
                if not tr:
                    logger.info("No data found for {}".format(
                        ".".join((net, site, loc, single_comp))))
                else:
                    if cache:
                        self.to_sds(tr)
            return Stream(tr)

        if not wildcard_request:
            # Original behavior: single component Trace
            tr = _fetch_single_component(
                component_codes[0], start, end)
            return tr
        else:
            stream = Stream()
            for c_code in component_codes:
                trc = _fetch_single_component(c_code, start, end)
                if trc:
                    stream += trc
            if len(stream) == 0:
                logger.error(
                    "No components found for pattern {} in interval {} - {}".format(comp, start, end))
            return stream
