import os
from datetime import datetime
import pandas as pd
from zizou import DataSource, S3Waveforms, RSAM
from tonik import Storage
from obspy import UTCDateTime

from aitana.tilde import tilde_request
from aitana.wfs import wfs_request


class CraterLake(object):

    def __init__(self, startdate: datetime, enddate: datetime):
        self.startdate = startdate
        self.enddate = enddate

    def temperature(self, resample: str = "D", interpolate: str = None, exclude1995: bool = True, dropna: bool = True) -> pd.DataFrame:
        """
        Read crater lake temperature from Tilde (https://tilde.geonet.org.nz).

        Parameters:
        -----------
            :param enddate: The latest date of the time-series.
                            Mainly needed for testing.
            :type enddate: :class:`datetime.datetime`
            :param resample: Average over resampling interval and
                            linearly interpolate in between. The
                            interval should be one of 'D', 'W',
                            or 'M'
            :type resample: str
            :param interpolate: Interpolate between resampled points.
                                Only takes effect if resample is not
                                None.
            :type interpolate: str

        Returns:
        --------
            :param df: A dataframe with the temperature time series.
            :type df: :class:`pandas.Dataframe`
        """
        # read in datalogger temperature data
        df = tilde_request(startdate=max(self.startdate, datetime(2019, 5, 21, 4, 30, 0)), enddate=self.enddate,
                           domain="envirosensor",
                           name="lake-temperature",
                           station="RU001", sensor="04")

        df1 = tilde_request(startdate=max(self.startdate, datetime(1993, 12, 11)), enddate=self.enddate,
                            domain="envirosensor",
                            name="lake-temperature",
                            station="RU001", sensor="01")

        # Read in the manual temperatures
        # water analysis lab, thermometer
        dfmc1 = tilde_request(startdate=max(self.startdate, datetime(1954, 2, 14)), enddate=self.enddate,
                              domain="manualcollect",
                              name="lake-temperature",
                              station="RU001",
                              method="thermometer",
                              sensor="MC01")

        # water analysis lab, thermocouple
        dfmc1_2 = tilde_request(startdate=max(self.startdate, datetime(1991, 1, 13)), enddate=self.enddate,
                                domain="manualcollect",
                                name="lake-temperature",
                                station="RU001",
                                method="thermocouple",
                                sensor="MC01")

        # thermocouple 1
        dfmc3 = tilde_request(startdate=max(self.startdate, datetime(1998, 3, 17)), enddate=self.enddate,
                              domain="manualcollect",
                              name="lake-temperature",
                              station="RU001",
                              method="thermocouple",
                              sensor="MC03")

        # thermocouple 2
        dfmc4 = tilde_request(startdate=max(self.startdate, datetime(1991, 1, 13)), enddate=self.enddate,
                              domain="manualcollect",
                              name="lake-temperature",
                              station="RU001",
                              method="thermocouple",
                              sensor="MC04")

        df = df.combine_first(df1)
        df = df.combine_first(dfmc1)
        df = df.combine_first(dfmc1_2)
        df = df.combine_first(dfmc3)
        df = df.combine_first(dfmc4)

        if resample is not None:
            df = df.resample("D").mean()
            if interpolate is not None:
                df = df.interpolate(interpolate)

        # remove observations following the 1995 eruption when
        # the measured temperatures do not represent a lake.
        if exclude1995:
            cond1 = df.index <= "1995-09-20 00:00:00"
            cond2 = df.index >= "2000-01-01 00:00:00"
            df = df[(cond1) | (cond2)]
        df = df[df.index <= str(self.enddate)]
        if self.startdate is not None:
            df = df[df.index >= str(self.startdate)]

        # get rid of nans
        if dropna:
            df.dropna(inplace=True)

        # change annoying temp label and drop old column
        return df

    def water_analyte(self, analyte: str, resample: str = None, interpolate: str = None,
                      exclude1995: bool = True) -> pd.DataFrame:
        """
        Download water analyte data from Tilde (https://tilde.geonet.org.nz).

        Parameters:
        -----------
            analyte : str
                The analyte to download. This can be one of
                    'Al, 'As', 'B', 'Br', 'Ca', 'Cl', 'Cs', 'F',
                    'Fe', 'H2S', 'K', 'Li', 'Mg', 'NH3', 'NO3-N', 'Na',
                    'PO4-P', 'Rb', 'SO4', 'SiO2', 'd18O', 'd2H', 'ph'
            resample : str
                Resample the data to a given interval. This can be anything
                allowed by :pandas.DataFrame.resample:. 
            interpolate : str
                Interpolate between points. This can be anything allowed by
                :pandas.DataFrame.interpolate:.
            exclude1995 : bool
                Exclude data from 1995 when there wasn't a lake.
        Returns:
        --------
            pandas.DataFrame
                A dataframe with the water analyte time series.
        """
        df = tilde_request(startdate=max(self.startdate, datetime(1964, 5, 9)),
                           enddate=self.enddate,
                           domain="manualcollect",
                           name=f"lake-{analyte}-conc",
                           station="RU001",
                           sensor="MC01")
        df1 = tilde_request(startdate=max(self.startdate, datetime(1964, 5, 9)),
                            enddate=self.enddate,
                            domain="manualcollect",
                            name=f"lake-{analyte}-conc",
                            station="RU001",
                            sensor="MC03")
        df2 = tilde_request(startdate=max(self.startdate, datetime(1964, 5, 9)),
                            enddate=self.enddate,
                            domain="manualcollect",
                            name=f"lake-{analyte}-conc",
                            station="RU001",
                            sensor="MC04")
        df = df.combine_first(df1)
        df = df.combine_first(df2)
        if resample is not None:
            df = df.resample(resample).mean()

        if interpolate is not None:
            df = df.interpolate(interpolate)

        if exclude1995:
            cond1 = df.index <= "1995-09-20 00:00:00"
            cond2 = df.index >= "2000-01-01 00:00:00"
            df = df[(cond1) | (cond2)]

        df = df[df.index <= str(self.enddate)]
        if self.startdate is not None:
            df = df[df.index >= str(self.startdate)]
        return df


class Gas(object):

    def __init__(self, startdate: datetime, enddate: datetime):
        self.startdate = startdate
        self.enddate = enddate

    def so2(self) -> pd.DataFrame:
        df_cospec = tilde_request(startdate=max(self.startdate, datetime(2003, 5, 27)),
                                  enddate=self.enddate,
                                  domain="manualcollect",
                                  name="plume-SO2-gasflux",
                                  method="cospec",
                                  station="RU000", sensor="MC01")
        df_contouring = tilde_request(startdate=max(self.startdate, datetime(2004, 4, 21)),
                                      enddate=self.enddate,
                                      domain="manualcollect",
                                      name="plume-SO2-gasflux",
                                      method="contouring",
                                      station="RU000", sensor="MC01")
        df_flyspec = tilde_request(startdate=max(self.startdate, datetime(2008, 8, 11)),
                                   enddate=self.enddate,
                                   domain="manualcollect",
                                   name="plume-SO2-gasflux",
                                   method="flyspec",
                                   station="RU000", sensor="MC01")
        df_mobiledoas = tilde_request(startdate=max(self.startdate, datetime(2023, 8, 23)),
                                      enddate=self.enddate,
                                      domain="manualcollect",
                                      name="plume-SO2-gasflux",
                                      method="mobile-doas",
                                      station="RU000", sensor="MC01")
        df = df_cospec
        df = df.combine_first(df_contouring)
        df = df.combine_first(df_flyspec)
        df = df.combine_first(df_mobiledoas)
        df['obs'] *= 86.4
        df['err'] *= 86.4
        df = df[df.index <= str(self.enddate)]
        if self.startdate is not None:
            df = df[df.index >= str(self.startdate)]
        return df

    def co2(self) -> pd.DataFrame:
        df = tilde_request(startdate=max(self.startdate, datetime(2003, 5, 27)),
                           enddate=self.enddate,
                           domain="manualcollect",
                           name="plume-CO2-gasflux",
                           method="contouring",
                           station="RU000", sensor="MC01")
        df['obs'] *= 86.4
        df['err'] *= 86.4
        df = df[df.index <= str(self.enddate)]
        if self.startdate is not None:
            df = df[df.index >= str(self.startdate)]
        return df

    def h2s(self) -> pd.DataFrame:
        df = tilde_request(startdate=max(self.startdate, datetime(2004, 4, 21)),
                           enddate=self.enddate,
                           domain="manualcollect",
                           name="plume-H2S-gasflux",
                           method="contouring",
                           station="RU000", sensor="MC01")
        df['obs'] *= 86.4
        df['err'] *= 86.4
        df = df[df.index <= str(self.enddate)]
        if self.startdate is not None:
            df = df[df.index >= str(self.startdate)]
        return df


class Seismicity(object):

    def __init__(self, startdate: datetime, enddate: datetime):
        self.startdate = startdate
        self.enddate = enddate
        self.cache_dir = os.path.join(os.environ.get(
            "HOME", "/tmp"), "aitana_cache")
        self.feature_dir = os.path.join(os.environ.get(
            "HOME", "/tmp"), "aitana_features")

    def regional(self):
        """
        Load an earthquake catalogue for an area containing Waiouru and National Park.
        """

        polygon = (
            "175.32+-39.50,+175.32+-39.18,+175.77+-39.18,+175.77+-39.50,+175.32+-39.50"
        )
        return wfs_request(self.startdate, self.enddate, polygon=polygon)

    def cone(self):
        """
        Load an earthquake catalogue for an area of 7 km around the summit.
        """
        radius = 7000
        center_point = "175.57+-39.28"
        return wfs_request(self.startdate, self.enddate, radius=radius, center_point=center_point)

    def rm_duplicates(self, cat1: pd.DataFrame, cat2: pd.DataFrame) -> pd.DataFrame:
        """
        Remove events from catalogue cat1 that are also in cat2 and return a new catalogue.

        Parameters:
        -----------
            cat1 : pandas.DataFrame
                The first catalogue.
            cat2 : pandas.DataFrame
                The second catalogue.

        Returns:
        --------
            pandas.DataFrame
                A new catalogue with the events from cat1 that are not in cat2.
        """
        idxs = []
        cat_tmp = cat1.copy()
        for i in range(cat_tmp.shape[0]):
            if cat_tmp.iloc[i].publicid in cat2.publicid.values:
                idxs.append(i)
        cat_tmp.drop(idxs, inplace=True)
        return cat_tmp

    def rsam(self, station: str = "NZ.FWVZ.10.HHZ") -> pd.DataFrame:
        """
        Compute RSAM for a given station.

        Parameters:
        -----------
            station : str
                The station to compute RSAM for in the form "NET.STA.LOC.CHA".

        Returns:
        --------
            pandas.DataFrame
                A dataframe with the RSAM values.
        """
        fdsnurls = ["https://service.geonet.org.nz",
                    "https://service-nrt.geonet.org.nz"]
        s3bucket = "geonet-open-data"
        fout = os.path.join(
            self.feature_dir, f"{station}_rsam_{str(self.startdate)}_{str(self.enddate)}.csv")
        if os.path.exists(fout):
            return pd.read_csv(fout, index_col=0, parse_dates=True)
        s3client = S3Waveforms(s3bucket, fdsnurls, staxml_dir=self.feature_dir)
        ds = DataSource(clients=[s3client],
                        chunk_size=86400, cache_dir=self.cache_dir)
        net, sta, loc, cha = station.split(".")
        r = RSAM(interval=600)
        s = Storage('aitana_rsam', rootdir=self.feature_dir,
                    starttime=self.startdate, endtime=self.enddate)
        store = s.get_substore(net, sta, loc, cha)
        for trace in ds.get_waveforms(net, sta, loc, cha, UTCDateTime(self.startdate),
                                      UTCDateTime(self.enddate), cache=True):
            if not trace:
                continue
            else:
                print(trace)
            xds = r.compute(trace)
            store.save(xds)
        rsam_xds = store('rsam')
        rsam_df = rsam_xds.to_dataframe()
        rsam_df.to_csv(fout)
        return rsam_df


def eruptions(min_size: int = 0, dec_interval: str = None) -> pd.DataFrame:
    """
    This function loads the eruption catalogue for Mt. Ruapehu and declusters
    it if required. The catalogue is based on Brad Scott's GNS Science Report.
    Declustering is turned on if dec_interval is greater than zero. This will
    also exclude magmatic eruption episodes between 9 January 1944 and
    8 January 1946 as well as 6 January 1995 to 12 January 1997 as these aren't
    independent events.

    Parameters:
    -----------
        min_size : int
            The minimum eruption size to include in the catalogue.
        dec_interval : str 
            The declustering interval in pandas complient time delta format.

    Returns:
    --------
        pandas.DataFrame
            A dataframe with the (declustered) eruption catalogue.
    """
    filename = "https://raw.githubusercontent.com/GeoNet/data/refs/heads/main/"
    filename += "historic-volcanic-activity/historic_eruptive_activity_ruapehu.csv"
    df = pd.read_csv(filename, index_col=0, parse_dates=True, comment='#')

    # Select eruptions of a particular scale or larger
    scaled_eruptions = df[df['Activity Scale'] >= min_size].copy()

    if dec_interval is not None:
        # duplicate time index as a data column
        scaled_eruptions.insert(1, "tvalue", scaled_eruptions.index)
        # calculate intereruption event time
        delta = (scaled_eruptions["tvalue"] - scaled_eruptions["tvalue"].shift()).fillna(
            pd.Timedelta(seconds=0)
        )
        scaled_eruptions.insert(1, "delta", delta)
        scaled_eruptions.iloc[0, 1] = pd.Timedelta(dec_interval)
        # Exclude certain date ranges from calculations to ensure a more
        # Poisson-like process by removing long term eruption periods.
        period1 = scaled_eruptions.index < "1944-10-01 00:00:00"
        period2 = scaled_eruptions.index > "1946-08-01 00:00:00"
        scaled_eruptions = scaled_eruptions[period1 | period2]
        period3 = scaled_eruptions.index < "1995-07-01 00:00:00"
        period4 = scaled_eruptions.index > "1997-12-01 00:00:00"
        scaled_eruptions = scaled_eruptions[period3 | period4]
        scaled_eruptions = scaled_eruptions[(
            scaled_eruptions.delta >= dec_interval)]
        scaled_eruptions = scaled_eruptions.drop(columns=["delta", "tvalue"])
    return scaled_eruptions
