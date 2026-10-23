import inspect
import logging
import os
import tempfile
from datetime import timedelta

import fsspec
import numpy as np
from aitana.testing import synthetic_trace
from obspy import Inventory, Stream, Trace, UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client as FDSN_Client
from obspy.clients.filesystem.sds import Client as SDS_Client

logger = logging.getLogger(__name__)

STAXML_DIR = os.environ.get(
    "AITANA_STAXML_DIR",
    os.path.join(os.environ["HOME"], ".aitana-access-instrument-response"),
)


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
                "No post processing not set up for this case:\n{}".format(self.st)
            )
        elif sum(self.res) > 1:
            msg = (
                "More than one check is true: {}. Processing workflow "
                "is unclear.".format(
                    ", ".join([a[0] for (a, b) in zip(self.checklist, self.res) if b])
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
                "Post-procesing function not equipped tohandle:\n{}".format(self.st)
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
    fdsn_urls=("https://service.geonet.org.nz"),
):
    """
    Retrieve instrument response file in STATIONXML format.
    """
    fn = "{:s}.xml".format(site)
    fn = os.path.join(STAXML_DIR, fn)
    try:
        inv = read_inventory(fn, format="STATIONXML")
    except FileNotFoundError:
        try:
            os.makedirs(STAXML_DIR, exist_ok=True)
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
    """
    Base class for waveform clients.

    Clients implement `get_raw`. Sensitivity removal, gap filling and
    trimming happen in `process`, so that `SeismicData` can apply them to
    raw data read from its cache as well.
    """

    def get_raw(self, net, site, loc, comp, start, end):
        """
        Get waveform data as delivered by the source.

        Parameters
        ----------
        net, site, loc, comp : str
            Network, station, location and channel codes.
        start, end : UTCDateTime
            Start and end of the requested time span.

        Returns
        -------
        Stream
            Unmerged waveform data in counts.
        """
        raise NotImplementedError

    def process(self, st, net, site, loc, comp, start, end):
        """
        Remove the instrument sensitivity, fill gaps and trim to the
        requested time span.

        Parameters
        ----------
        st : Stream
            Raw waveform data as returned by `get_raw`.
        net, site, loc, comp : str
            Network, station, location and channel codes.
        start, end : UTCDateTime
            Start and end of the requested time span.

        Returns
        -------
        Trace
            Waveform data with the instrument sensitivity removed.
        """
        inv = get_instrument_response(net, site, loc, fdsn_urls=self.fdsn_urls)
        inv_dt = inv.select(
            location=loc, channel=comp, starttime=start, endtime=end - 1
        )
        # Integer counts can't hold NaN as fill value. Not every PostProcess
        # output merges, and they all return only the first trace.
        for tr in st:
            tr.data = tr.data.astype(np.float64)
        st.merge(fill_value=self.fill_value)

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
        return pp.run_post_processing()

    def get_waveforms(self, net, site, loc, comp, start, end):
        """
        Get waveform data with the instrument sensitivity removed.

        Parameters
        ----------
        net, site, loc, comp : str
            Network, station, location and channel codes.
        start, end : UTCDateTime
            Start and end of the requested time span.

        Returns
        -------
        Trace
            Waveform data with the instrument sensitivity removed.
        """
        st = self.get_raw(net, site, loc, comp, start, end)
        return self.process(st, net, site, loc, comp, start, end)


class SDSWaveforms(WaveformBaseclass):
    def __init__(self, sds_dir, fdsn_urls, fill_value=np.nan):
        """
        Get seismic waveforms from a local SDS archive.

        :param fill_value: Default value used to fill gaps and pad traces to
                           cover the whole time period defined by `startdate`
                           and `enddate`. If 'interpolate' is chosen, the
                           traces will not be padded.
        :type fill_value: int, float, 'interpolate', np.nan, or None
        """
        self.client = SDS_Client(sds_dir)
        self.fdsn_urls = fdsn_urls
        self.fill_value = fill_value

    def get_raw(self, net, site, loc, comp, startdate, enddate):
        return self.client.get_waveforms(net, site, loc, comp, startdate, enddate)


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
        tr = synthetic_trace(
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

    def get_raw(self, net, site, loc, comp, startdate, enddate):
        """
        Generate synthetic daily files that start at midnight on the
        startdate and end at midnight on the enddate. The return the requested
        timespan from these files.
        """
        self.generate_dataset(
            self.client.sds_root, net, site, loc, comp, startdate, enddate
        )
        return self.client.get_waveforms(
            net, site, loc, comp, startdate, enddate, dtype="float64"
        )

    def process(self, st, net, site, loc, comp, startdate, enddate):
        """
        Return the first trace unchanged, since synthetic data have no
        station metadata.

        Parameters
        ----------
        st : Stream
            Synthetic waveform data as returned by `get_raw`.
        net, site, loc, comp : str
            Network, station, location and channel codes.
        startdate, enddate : UTCDateTime
            Start and end of the requested time span.

        Returns
        -------
        Trace
            First trace of `st`.
        """
        return st[0]


class FDSNWaveforms(WaveformBaseclass):
    def __init__(self, url, debug=False, fill_value=np.nan):
        """
        Get seismic waveforms from FDSN web service.
        """
        self.client = FDSN_Client(base_url=url, debug=debug)
        self.fdsn_urls = (self.client.base_url,)
        self.fill_value = fill_value

    def get_raw(self, net, site, loc, comp, startdate, enddate):
        return self.client.get_waveforms(net, site, loc, comp, startdate, enddate)


class S3Waveforms(WaveformBaseclass):
    def __init__(self, debug=False, fill_value=np.nan):
        """
        Get seismic waveforms from GeoNet's open data archive on AWS S3.

        Parameters
        ----------
        debug : bool
            If True, debug messages will be printed to the console.
        fill_value : int, float, 'interpolate', np.nan, or None
            Value to use for filling gaps in the data.
        """
        self.fsclient = fsspec.filesystem("s3", anon=True)
        self.fdsn_urls = [
            "https://service.geonet.org.nz",
            "https://service-nrt.geonet.org.nz",
        ]
        self.debug = debug
        self.fill_value = fill_value
        self.bucket = "geonet-open-data"

    def get_raw(self, net, site, loc, comp, start, end):
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
        Stream
            The day files' waveform data in counts, trimmed to the requested
            time span.
        """
        PATH_FORMAT = (
            "{bucket}/waveforms/miniseed/{year}/{year}.{julday:03d}/{station}."
        )
        PATH_FORMAT += "{network}/{year}.{julday:03d}.{station}."
        PATH_FORMAT += "{location}-{channel}.{network}.D"
        t_start = start
        t_end = UTCDateTime(year=t_start.year, julday=t_start.julday) + timedelta(days=1)
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
                _st = read(f)
            _st.trim(t_start, t_end, nearest_sample=False)
            st += _st
            t_start = t_end
            t_end = t_start + timedelta(days=1)
            t_end = min(t_end, end)
        return st


class SeismicData:
    def __init__(
        self, clients, chunk_size=None, cache_dir=None, cache_min_age=2 * 86400.0
    ):
        """
        Get seismic waveforms from the first of several clients that has
        data, optionally through a local cache of raw day files.

        Parameters
        ----------
        clients : list of WaveformBaseclass
            Waveform clients, tried in order.
        chunk_size : float, optional
            Length in seconds of the chunks `get_waveforms` yields.
        cache_dir : str, optional
            Parent directory of the cache, by default ``~/aitana_cache``.
            Raw day files go into its ``raw`` subdirectory in SDS layout.
            Files directly in ``cache_dir`` are processed data written by
            earlier versions and are never read.
        cache_min_age : float
            Seconds after the end of a UTC day before it is cached.
        """
        self.chunk_size = chunk_size

        if len(clients) < 1:
            msg = "At least one data client is required."
            raise ValueError(msg)

        # Test that all waveform clients are derived from WaveformBaseclass
        for client in clients:
            if not issubclass(client.__class__, WaveformBaseclass):
                msg = "All clients must be derived from WaveformBaseclass."
                raise ValueError(msg)

        self.clients = clients
        if cache_dir is None:
            cache_dir = os.path.join(os.environ["HOME"], "aitana_cache")
        self.cache_dir = os.path.join(cache_dir, "raw")
        self.cache_min_age = cache_min_age

        os.makedirs(self.cache_dir, exist_ok=True)

    def to_sds(self, st: Stream, net, site, loc, comp, day):
        """
        Path of a day file in the cache, writing `st` to it if given.

        Parameters
        ----------
        st : Stream or None
            Raw data of one UTC day. If None, nothing is written.
        net, site, loc, comp : str
            Network, station, location and channel codes.
        day : UTCDateTime
            Midnight at the start of the day.

        Returns
        -------
        str
            Path of the day file.
        """
        sds_fmtstr = os.path.join(
            "{year}",
            "{network}",
            "{station}",
            "{channel}.{sds_type}",
            "{network}.{station}.{location}.{channel}.{sds_type}.{year}.{doy:03d}",
        )
        fullpath = sds_fmtstr.format(
            year=day.year,
            doy=day.julday,
            sds_type="D",
            network=net,
            station=site,
            location=loc,
            channel=comp,
        )
        fullpath = os.path.join(self.cache_dir, fullpath)
        if st is not None:
            dirname, _ = os.path.split(fullpath)
            os.makedirs(dirname, exist_ok=True)
            logger.debug(f"writing {st} to {fullpath}")
            # Concurrent readers must never see a partially written day
            fd, tmp = tempfile.mkstemp(dir=dirname)
            os.close(fd)
            st.write(tmp, format="MSEED")
            os.replace(tmp, fullpath)
        return fullpath

    def get_raw(self, net, site, loc, comp, start, end):
        """
        Get raw data per UTC day from the cache or from the first client
        that has data. Days that ended more than `cache_min_age` ago are
        fetched whole and written to the cache.

        Parameters
        ----------
        net, site, loc, comp : str
            Network, station, location and channel codes.
        start, end : UTCDateTime
            Start and end of the requested time span.

        Returns
        -------
        st : Stream
            Raw data between `start` and `end`.
        client : WaveformBaseclass or None
            First client that delivered data, None if all of it came from
            the cache.
        """
        st = Stream()
        answered = None
        day = UTCDateTime(start.date)
        while day < end:
            path = self.to_sds(None, net, site, loc, comp, day)
            if os.path.exists(path):
                st += read(path)
                day += 86400
                continue
            cacheable = day + 86400 < UTCDateTime() - self.cache_min_age
            if cacheable:
                t0, t1 = day, day + 86400
            else:
                t0, t1 = max(start, day), min(end, day + 86400)
            for client in self.clients:
                try:
                    _st = client.get_raw(net, site, loc, comp, t0, t1)
                except Exception as e:
                    logger.warning(
                        "{} failed to get waveforms for {} between {} "
                        "and {} ({!r}), trying next client.".format(
                            client.__class__.__name__,
                            ".".join((net, site, loc, comp)),
                            t0,
                            t1,
                            e,
                        )
                    )
                    continue
                if _st:
                    answered = answered or client
                    if cacheable:
                        self.to_sds(_st, net, site, loc, comp, day)
                    st += _st
                    break
            day += 86400
        st.trim(start, end, nearest_sample=False)
        return st, answered

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
            If True, raw data of UTC days that ended more than
            `cache_min_age` ago are read from, or written to, the local
            SDS cache.

        Yields
        ------
        Trace or Stream
            For a single component request, yields an ObsPy ``Trace`` (backwards
            compatible). For a wildcard multi-component request, yields an
            ObsPy ``Stream`` with one Trace per component for each chunk.
        """
        # Detect simple wildcard request
        wildcard_request = ("?" in comp) or ("*" in comp)
        component_codes = [comp]
        if wildcard_request:
            # Only support patterns where wildcard replaces exactly one character
            # Typical use-case: 'HH?' or 'BH?'. Expand into Z,N,E first; if no
            # data found for N/E try 1/2 as horizontal components.
            base = comp.replace("*", "?")  # treat * like ? for single-char
            if base.count("?") == 1:
                pre, post = base.split("?")
                candidate_sets = [["Z"], ["N", "E"], ["1", "2"]]
                expanded = []
                for candidates in candidate_sets:
                    expanded += [f"{pre}{c}{post}" for c in candidates]
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
            if cache:
                st, client = self.get_raw(net, site, loc, single_comp, t_start, t_end)
                if st:
                    try:
                        tr = (client or self.clients[0]).process(
                            st, net, site, loc, single_comp, t_start, t_end
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to process waveforms for {} between {} "
                            "and {} ({!r}).".format(
                                ".".join((net, site, loc, single_comp)),
                                t_start,
                                t_end,
                                e,
                            )
                        )
                    else:
                        tr.stats["cached"] = client is None
            else:
                for client in self.clients:
                    try:
                        tr = client.get_waveforms(
                            net, site, loc, single_comp, t_start, t_end
                        )
                    except Exception as e:
                        logger.warning(
                            "{} failed to get waveforms for {} between {} "
                            "and {} ({!r}), trying next client.".format(
                                client.__class__.__name__,
                                ".".join((net, site, loc, single_comp)),
                                t_start,
                                t_end,
                                e,
                            )
                        )
                        continue
                    else:
                        break
            if not tr:
                logger.info(
                    "No data found for {}".format(
                        ".".join((net, site, loc, single_comp))
                    )
                )
            return tr

        t_start = start
        t_end = end
        if self.chunk_size is not None:
            t_end = t_start + self.chunk_size
            t_end = min(t_end, end)

        while t_end <= end:
            if not wildcard_request:
                # Original behavior: single component Trace
                tr = _fetch_single_component(component_codes[0], t_start, t_end)
                yield tr
            else:
                stream = Stream()
                for c_code in component_codes:
                    trc = _fetch_single_component(c_code, t_start, t_end)
                    if trc:
                        stream += trc
                if len(stream) == 0:
                    logger.error(
                        "No components found for pattern {} in interval {} - {}".format(
                            comp, t_start, t_end
                        )
                    )
                yield stream

            # Chunk progression logic (unchanged for both modes)
            if self.chunk_size is not None:
                t_start = t_end
                cond1 = (end - t_end) < self.chunk_size
                cond2 = end > t_end
                if cond1 and cond2:
                    last_chunk = end - t_end
                    t_end = t_start + last_chunk
                else:
                    t_end = t_start + self.chunk_size
            else:
                t_end = end + 1.0


if __name__ == "__main__":
    import doctest

    doctest.testmod()
