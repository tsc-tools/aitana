from datetime import datetime, timezone
import os

import numpy as np
import pandas as pd
import requests

from aitana import get_data
from aitana.tilde import tilde_request
from aitana.assimilate import SO2FusionModel
from aitana.util import cache_dataframe, eqRate, reindex
from aitana.wfs import wfs_request


class Gas(object):

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.end_date = pd.to_datetime(end_date, utc=True)

    @cache_dataframe()
    def so2(self, fuse: bool = False, smooth: bool = True) -> pd.DataFrame:

        dataframes = {}
        method = "cospec"
        if self.end_date > datetime(2023, 11, 9, 23, 1, 0, tzinfo=timezone.utc):
            method = "contouring"
        try:
            df = tilde_request(start_date=datetime(2000, 1, 1),
                               end_date=self.end_date,
                               domain="manualcollect",
                               name="plume-SO2-gasflux",
                               method=method,
                               station="WI000",
                               sensor="MC01")
        except ValueError:
            df = pd.DataFrame()
        else:
            dataframes['WI000'] = df

        if not fuse:
            return df * 86.4

        # Note: north-east point (WID01) is reporting consistently too low values
        for station in ["WID01", "WID02"]:
            try:
                df_ = tilde_request(start_date=datetime(2000, 1, 1),
                                    end_date=self.end_date,
                                    domain="scandoas",
                                    name="gasflux",
                                    method="reviewed",
                                    station=station,
                                    sensor="01")
            except ValueError:
                continue
            else:
                dataframes[station] = df_

        startdate = np.array([df.index.min()
                             for df in list(dataframes.values())]).min()
        enddate = np.array([df.index.max()
                           for df in list(dataframes.values())]).max()
        dates = pd.date_range(startdate, enddate, freq="1D")
        gasdata = {}
        interval = "1D"
        for key, df in dataframes.items():
            gasdata[key] = (
                df["obs"].groupby(pd.Grouper(freq=interval)
                                  ).median().reindex(dates).values
            )

        gasdata = pd.DataFrame(gasdata, index=dates)

        # define a covariance matrix for the observations that gives cospec values
        # a higher weight than mdoas values
        obs_cov = np.diag(np.array([5., 200., 20.])**2)
        model = SO2FusionModel(measurements=gasdata, initial_state=np.array([0]),
                               initial_cov=np.diag([[1]]),
                               obs_cov=obs_cov, k_states=1, k_posdef=1)
        # Set initial parameters for process noise
        initial_params = [0.5]
        # Fit the model
        result = model.fit(start_params=initial_params, disp=False)

        # Get the smoothed state estimates (filtered values)
        filtered_state = result.filtered_state[0]
        smoothed_state = result.smoothed_state[0]

        mean_trace = filtered_state
        error = np.sqrt(result.filtered_state_cov[0, 0])
        if smooth:
            mean_trace = smoothed_state
            error = np.sqrt(result.smoothed_state_cov[0, 0])
        so2df = pd.DataFrame(
            {"obs": mean_trace, "err": error}, index=gasdata.index)
        so2df.index.name = "dt"
        return so2df * 86.4

    @cache_dataframe()
    def co2(self) -> pd.DataFrame:
        df = tilde_request(start_date=datetime(2000, 1, 1),
                           end_date=self.end_date,
                           domain="manualcollect",
                           name="plume-CO2-gasflux",
                           method="contouring",
                           station="WI000", sensor="MC01")
        return df * 86.4

    @cache_dataframe()
    def h2s(self) -> pd.DataFrame:
        df = tilde_request(start_date=datetime(2000, 1, 1),
                           end_date=self.end_date,
                           domain="manualcollect",
                           name="plume-H2S-gasflux",
                           method="contouring",
                           station="WI000", sensor="MC01")
        return df * 86.4


class Seismicity(object):

    def __init__(self, start_date: datetime, end_date: datetime):
        self.start_date = pd.to_datetime(start_date, utc=True)
        self.end_date = pd.to_datetime(end_date, utc=True)

    def rsam(self) -> pd.DataFrame:
        """
        Load RSAM values NZ.WIZ.10.HHZ.The first time you call
        this it will download the data from Zenodo and store it
        locally.

        Returns:
        --------
            :param df: Dataframe with RSAM values.
            :type df: :class:`pandas.DataFrame`
        """
        rsam_fn = get_data('data/RSAM_NZ.WIZ.10.HHZ.csv')
        if not os.path.isfile(rsam_fn):
            # download the data from zenodo
            print('Downloading RSAM data from Zenodo')
            url = "https://zenodo.org/records/14759090/files/RSAM_NZ.WIZ.10.HHZ.csv"
            response = requests.get(url)
            if response.status_code == 200:
                with open(rsam_fn, "wb") as f:
                    f.write(response.content)
                print('Successfully downloaded RSAM data from Zenodo')
            else:
                print('Failed to download RSAM data from Zenodo')

        df = pd.read_csv(rsam_fn, skiprows=1, parse_dates=True, names=['dt', 'obs'], index_col=0,
                         date_format='ISO8601')
        if self.end_date.tzinfo is not None:
            df.index = df.index.tz_localize(timezone.utc)
        df = df[df.index <= str(self.end_date)]
        df = df[df.index >= str(self.start_date)]
        return df

    @cache_dataframe()
    def quakes(self):
        center_point = "177.186833+-37.523118"
        return wfs_request(self.start_date, self.end_date, radius=20000,
                           center_point=center_point, maxdepth=30)


@cache_dataframe()
def eruptions(min_size: int = 0, dec_interval: str = None) -> pd.DataFrame:
    """
    This function loads the eruption catalogue for White Island and
    declusters it.

    :param eruption_scale: The minimum eruption scale (between 0 and 4)
                           to use. This is currently ignored as the
                           White Island catalogue does not contain
                           eruption scale.
    :type eruption_scale: int
    :param dec_interval: The declustering interval, which is the minimum
                         distance in time between any two eruptions.
    :type dec_interval: :class:`pandas.DateOffset` or str
    :param datadir: Path that contains the catalogue file in csv format.
    :type datadir: str
    :returns: Declustered eruption catalogue
    :rtype: :class:`pandas.DataFrame`
    """
    eruptions = pd.read_csv(
        get_data("data/WhiteIs_Eruption_Catalogue.csv"),
        index_col="Date",
        parse_dates=True,
        comment="#",
    )

    # Select eruptions of a particular scale or larger
    eruptions = eruptions[eruptions.Activity_Scale >= min_size].copy()

    if dec_interval is not None:
        # duplicate time index as a data column
        eruptions.insert(1, "tvalue", eruptions.index)
        # calculate intereruption event time
        delta = (eruptions["tvalue"] - eruptions["tvalue"].shift()).fillna(
            pd.Timedelta(seconds=0)
        )
        eruptions.insert(1, "delta", delta)
        eruptions.iloc[0, 1] = pd.Timedelta(dec_interval)
        eruptions = eruptions[(eruptions.delta >= dec_interval)]
        eruptions = eruptions.drop(columns=["tvalue", "delta"])

    eruptions.index = pd.DatetimeIndex(eruptions.index)
    eruptions = eruptions.tz_localize('utc')
    return eruptions


@cache_dataframe()
def load_all(
    fill_method="interpolate",
    start_date=datetime(2005, 1, 1, tzinfo=timezone.utc),
    end_date=datetime.now(timezone.utc),
    ignore_data=(),
    fuse_so2=False,
):
    cols = {}
    s = Seismicity(start_date, end_date)
    cat = s.quakes()
    dft = eqRate(cat, fixed_time=7).resample("D").mean().interpolate()
    new_dates = pd.date_range(dft.index[0], end_date, freq="D")
    if "Eqr" not in ignore_data:
        cols["Eqr"] = reindex(dft, new_dates, fill_method=fill_method)
    if "RSAM" not in ignore_data:
        rsam = s.rsam()
        cols["RSAM"] = reindex(rsam, new_dates, fill_method=fill_method)

    g = Gas(start_date, end_date)
    if "CO2" not in ignore_data:
        co2 = g.co2()
        cols["CO2"] = reindex(co2, new_dates, fill_method=fill_method)
    if "H2S" not in ignore_data:
        h2s = g.h2s()
        cols["H2S"] = reindex(h2s, new_dates, fill_method=fill_method)
    if "SO2" not in ignore_data:
        smooth = False
        if fill_method == "interpolate":
            smooth = True
            fuse_so2 = True
        so2 = g.so2(fuse=fuse_so2, smooth=smooth)
        cols["SO2"] = reindex(so2, new_dates, fill_method=fill_method)
    rdf = pd.DataFrame(cols, index=new_dates)
    return rdf
