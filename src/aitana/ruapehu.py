import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from aitana.tilde import tilde_request
from aitana.wfs import wfs_request
from aitana import get_data
from aitana.util import cache_dataframe, gradient, reindex, eqRate


class CraterLake(object):

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date

    @cache_dataframe()
    def temperature(self, resample: str = "D", interpolate: str = None,
                    exclude1995: bool = True, dropna: bool = True) -> pd.DataFrame:
        """
        Read crater lake temperature from Tilde (https://tilde.geonet.org.nz).

        Parameters:
        -----------
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
        dataframes = []
        # read in datalogger temperature data
        try:
            df = tilde_request(start_date=datetime(2019, 5, 21, 4, 30, 0), end_date=self.end_date,
                               domain="envirosensor",
                               name="lake-temperature",
                               station="RU001", sensor="04")
        except ValueError:
            pass
        else:
            dataframes.append(df)

        try:
            df1 = tilde_request(start_date=datetime(1993, 12, 11), end_date=self.end_date,
                                domain="envirosensor",
                                name="lake-temperature",
                                station="RU001", sensor="01")
        except ValueError:
            pass
        else:
            dataframes.append(df1)

        # Read in the manual temperatures
        # water analysis lab, thermometer
        try:
            dfmc1 = tilde_request(start_date=datetime(1954, 2, 14), end_date=self.end_date,
                                  domain="manualcollect",
                                  name="lake-temperature",
                                  station="RU001",
                                  method="thermometer",
                                  sensor="MC01")
        except ValueError:
            pass
        else:
            if len(dfmc1) > 0:
                dataframes.append(dfmc1)

        # water analysis lab, thermocouple
        try:
            dfmc1_2 = tilde_request(start_date=datetime(1991, 1, 13), end_date=self.end_date,
                                    domain="manualcollect",
                                    name="lake-temperature",
                                    station="RU001",
                                    method="thermocouple",
                                    sensor="MC01")
        except ValueError:
            pass
        else:
            if len(dfmc1_2) > 0:
                dataframes.append(dfmc1_2)

        # thermocouple 1
        try:
            dfmc3 = tilde_request(start_date=datetime(1998, 3, 17), end_date=self.end_date,
                                  domain="manualcollect",
                                  name="lake-temperature",
                                  station="RU001",
                                  method="thermocouple",
                                  sensor="MC03")
        except ValueError:
            pass
        else:
            if len(dfmc3) > 0:
                dataframes.append(dfmc3)

        # thermocouple 2
        try:
            dfmc4 = tilde_request(start_date=datetime(1991, 1, 13), end_date=self.end_date,
                                  domain="manualcollect",
                                  name="lake-temperature",
                                  station="RU001",
                                  method="thermocouple",
                                  sensor="MC04")
        except ValueError:
            pass
        else:
            if len(dfmc4) > 0:
                dataframes.append(dfmc4)

        if len(dataframes) == 0:
            raise ValueError(
                f"No data found for lake temperature between {self.start_date} and {self.end_date}")

        df = dataframes[0]
        if len(dataframes) > 1:
            for df1 in dataframes[1:]:
                df = df.combine_first(df1)

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
        # get rid of nans
        if dropna:
            df.dropna(inplace=True)
        # change annoying temp label and drop old column
        df.rename(columns={" t (C)": "obs"}, inplace=True)
        return df

    @cache_dataframe()
    def water_level(self):
        """
        Read crater lake water level from Tilde (https://tilde.geonet.org.nz).
        """
        sensors = ['02', '03']
        dataframes = []
        for sensor in sensors:
            try:
                df = tilde_request(start_date=datetime(2009, 4, 15, 2, 0, 0), end_date=self.end_date,
                                   domain="envirosensor",
                                   name="lake-height",
                                   station="RU001", sensor=sensor)
            except ValueError:
                continue
            if len(df) > 0:
                dataframes.append(df)

        df = dataframes[0]
        if len(dataframes) > 1:
            for df1 in dataframes[1:]:
                df = df.combine_first(df1)
        return df

    @cache_dataframe()
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
        dataframes = []
        sensors = ["MC03", "MC04", "MC01"]
        for sensor in sensors:
            try:
                df = tilde_request(start_date=datetime(1964, 5, 9),
                                   end_date=self.end_date,
                                   domain="manualcollect",
                                   name=f"lake-{analyte}-conc",
                                   station="RU001",
                                   sensor=sensor)
            except ValueError:
                continue
            if len(df) > 0:
                dataframes.append(df)
        if len(dataframes) == 0:
            raise ValueError(
                f"No data found for {analyte} between {self.start_date} and {self.end_date}")
        df = dataframes[0]
        if len(dataframes) > 1:
            for df1 in dataframes[1:]:
                df = df.combine_first(df1)
        if resample is not None:
            df = df.resample(resample).mean()

        if interpolate is not None:
            df = df.interpolate(interpolate)

        if exclude1995:
            cond1 = df.index <= "1995-09-20 00:00:00"
            cond2 = df.index >= "2000-01-01 00:00:00"
            df = df[(cond1) | (cond2)]
        return df


class Gas(object):

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date

    @cache_dataframe()
    def so2(self) -> pd.DataFrame:
        dataframes = []
        for method in ["cospec", "contouring", "flyspec", "mobile-doas"]:
            try:
                df = tilde_request(start_date=datetime(2003, 5, 27),
                                   end_date=self.end_date,
                                   domain="manualcollect",
                                   name="plume-SO2-gasflux",
                                   method=method,
                                   station="RU000", sensor="MC01")
            except ValueError:
                continue
            if len(df) > 0:
                dataframes.append(df)
        if len(dataframes) == 0:
            raise ValueError(
                f"No data found for SO2 between {self.start_date} and {self.end_date}")
        df = dataframes[0]
        if len(dataframes) > 1:
            for df1 in dataframes[1:]:
                df = df.combine_first(df1)
        df['obs'] *= 86.4
        df['err'] *= 86.4
        return df

    @cache_dataframe()
    def co2(self) -> pd.DataFrame:
        df = tilde_request(start_date=max(self.start_date, datetime(2003, 5, 27)),
                           end_date=self.end_date,
                           domain="manualcollect",
                           name="plume-CO2-gasflux",
                           method="contouring",
                           station="RU000", sensor="MC01")
        df['obs'] *= 86.4
        df['err'] *= 86.4
        return df

    @cache_dataframe()
    def h2s(self) -> pd.DataFrame:
        df = tilde_request(start_date=max(self.start_date, datetime(2004, 4, 21)),
                           end_date=self.end_date,
                           domain="manualcollect",
                           name="plume-H2S-gasflux",
                           method="contouring",
                           station="RU000", sensor="MC01")
        df['obs'] *= 86.4
        df['err'] *= 86.4
        return df


class Seismicity(object):

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = start_date
        self.end_date = end_date

    @cache_dataframe()
    def regional(self):
        """
        Load an earthquake catalogue for an area containing Waiouru and National Park.
        """

        polygon = (
            "175.32+-39.50,+175.32+-39.18,+175.77+-39.18,+175.77+-39.50,+175.32+-39.50"
        )
        return wfs_request(self.start_date, self.end_date, polygon=polygon)

    @cache_dataframe()
    def cone(self):
        """
        Load an earthquake catalogue for an area of 7 km around the summit.
        """
        radius = 7000
        center_point = "175.57+-39.28"
        return wfs_request(self.start_date, self.end_date, radius=radius, center_point=center_point)

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
        dates_to_drop = cat_tmp.index[idxs]
        cat_tmp.drop(dates_to_drop, inplace=True)
        return cat_tmp

    def daily_rsam(self, update=False) -> pd.DataFrame:
        """
        Load RSAM values from DRZ, MAVZ and FWVZ combined by scaling
        MAVZ and FWVZ RSAM values with DRZ.

        Returns:
        --------
            :param df: Dataframe with RSAM values.
            :type df: :class:`pandas.DataFrame`
        """
        df = pd.read_csv(get_data("data/ruapehu_rsam.csv"),
                         parse_dates=True, index_col=0)
        if self.end_date.tzinfo is not None:
            df.index = df.index.tz_localize(timezone.utc)
        if self.end_date > df.index[-1] and update:
            df_drz = pd.read_csv(
                get_data("data/DRZ_scaled_MAVZ.csv"), parse_dates=True, index_col=0
            )
            df_drz.rename(columns={"RSAM": "obs"}, inplace=True)
            url = "http://kaizen.gns.cri.nz:9157/feature?name=rsam"
            url += "&starttime=2007-01-01T00:00:00"
            url += f"&endtime={self.end_date.isoformat().split('+')[0]}"
            url += "&volcano=Ruapehu&site=FWVZ"
            try:
                df_fwvz = pd.read_csv(
                    url,
                    parse_dates=True,
                    index_col=0,
                    date_format="ISO8601",
                )
            except Exception as e:
                print(url)
                raise e
            df_fwvz["rsam"] = np.where(
                df_fwvz["rsam"] > 1e30, np.nan, df_fwvz["rsam"])
            df_fwvz_daily = df_fwvz.resample("1D").mean()
            df_fwvz_daily_scaled = -2.3945 + 3.5062 * df_fwvz_daily
            df_fwvz_daily_scaled.rename(columns={"rsam": "obs"}, inplace=True)
            df = df_drz.combine_first(df_fwvz_daily_scaled)
            # remove duplicated dates:
            df = df.loc[~df.index.duplicated(), :]
            df.to_csv(get_data("data/ruapehu_rsam.csv"))
            if self.end_date.tzinfo is not None:
                df.index = df.index.tz_localize(timezone.utc)
        df = df[df.index <= str(self.end_date)]
        df = df[df.index >= str(self.start_date)]
        return df


@cache_dataframe()
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
    scaled_eruptions = scaled_eruptions.tz_localize('utc')
    return scaled_eruptions


@cache_dataframe()
def load_all(
    fill_method="interpolate",
    end_date=datetime.now(timezone.utc),
):
    end_date = end_date.replace(tzinfo=timezone.utc)
    cols = {}

    # load crater lake data
    rcl = CraterLake(start_date=datetime(1954, 1, 1), end_date=end_date)
    dft = rcl.temperature()
    grad = gradient(dft)
    new_dates = pd.date_range(dft.index[0], end_date, freq="D")
    cols["TemperatureBin"] = reindex(dft, new_dates, fill_method=fill_method)
    cols["GradientBin"] = reindex(grad, new_dates, fill_method=fill_method)
    mg = rcl.water_analyte("Mg")
    dmg = gradient(mg, period="90D")
    cols["Mg"] = reindex(dmg, new_dates, fill_method=fill_method)
    so4 = rcl.water_analyte("SO4")
    dso4 = gradient(so4, period="90D")
    cols["SO4"] = reindex(dso4, new_dates, fill_method=fill_method)
    cl = rcl.water_analyte("Cl")
    mg_cl = mg / cl
    dmg_cl = gradient(mg_cl, period="90D")
    cols["Mg_ClBin"] = reindex(dmg_cl, new_dates, fill_method=fill_method)
    na = rcl.water_analyte("Na")
    mg_na = mg / na
    dmg_na = gradient(mg_na, period="90D")
    cols["Mg_Na"] = reindex(dmg_na, new_dates, fill_method=fill_method)
    k = rcl.water_analyte("K")
    mg_k = mg / k
    dmg_k = gradient(mg_k, period="90D")
    cols["Mg_K"] = reindex(dmg_k, new_dates, fill_method=fill_method)
    al = rcl.water_analyte("Al")
    mg_al = mg / al
    dmg_al = gradient(mg_al, period="90D")
    cols["Mg_Al"] = reindex(dmg_al, new_dates, fill_method=fill_method)

    # Load gas data
    rg = Gas(start_date=datetime(1954, 1, 1), end_date=end_date)
    co2 = rg.co2()
    co2 = co2.groupby(pd.Grouper(freq="1D")).median()
    cols["CO2"] = reindex(co2, new_dates, fill_method=fill_method)
    so2 = rg.so2()
    so2 = so2.groupby(pd.Grouper(freq="1D")).median()
    cols["SO2"] = reindex(so2, new_dates, fill_method=fill_method)
    h2s = rg.h2s()
    h2s = h2s.groupby(pd.Grouper(freq="1D")).median()
    cols["H2S"] = reindex(h2s, new_dates, fill_method=fill_method)
    cs = co2 / so2
    dcs = gradient(cs, period="90D")
    cols["CO2_SO2"] = reindex(dcs, new_dates, fill_method=fill_method)

    # Load seismic data
    rs = Seismicity(start_date=datetime(2015, 1, 1), end_date=end_date)
    cat_outer = rs.regional()
    cat_inner = rs.cone()
    cat_outer = rs.rm_duplicates(cat_outer, cat_inner)
    eqr_outer = eqRate(cat_outer, fixed_time=7).resample(
        "D").mean().interpolate()
    eqr_inner = eqRate(cat_inner, fixed_time=7).resample(
        "D").mean().interpolate()
    cols["Eqr_outer"] = reindex(eqr_outer, new_dates, fill_method=fill_method)
    cols["Eqr_inner"] = reindex(eqr_inner, new_dates, fill_method=fill_method)
    rsam = rs.daily_rsam()
    cols["RSAM"] = reindex(rsam, new_dates, fill_method=fill_method)
    rsam100 = rsam.rolling(window="100D", min_periods=1).max()
    cols["RSAM100"] = reindex(rsam100, new_dates, fill_method=fill_method)
    return pd.DataFrame(cols, index=new_dates)
