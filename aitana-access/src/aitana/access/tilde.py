from datetime import timezone, datetime, timedelta
import io
import requests
import pandas as pd


def tilde_request(
    start_date: datetime,
    end_date: datetime,
    domain: str,
    name: str,
    station: str,
    method: str = '-',
    sensor: str = '-',
    aspect: str = '-',
) -> pd.DataFrame:
    """
    Request data from the tilde API (https://tilde.geonet.org.nz/v3/api-docs/).
    See the tilde discovery tool for more information:
    https://tilde.geonet.org.nz/ui/data-discovery/

    Parameters
    ----------
    domain : str
        The domain of the data (e.g. 'manualcollect')
    name : str
        The name of the data (e.g. 'plume-SO2-gasflux')
    station : str
        The station code (e.g. 'WI000')
    method : str
        The method of the data (e.g. 'contouring')
    sensor : str
        The sensor of the data (e.g. 'MC01')
    aspect : str
        The aspect of the data (e.g. 'nil')
    start_date : date
        The start date of the data
    end_date : date
        The end date of the data

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with the requested data
    """
    tilde_url = "https://tilde.geonet.org.nz/v4/data"
    # split the request into historic and latest data
    get_historic = True
    get_latest = False
    start_date = start_date.astimezone(timezone.utc)
    end_date = end_date.astimezone(timezone.utc)
    _tstart = str(start_date.date())
    _tend = str(end_date.date())
    _today = datetime.now(timezone.utc).date()
    if end_date.date() > (_today - timedelta(days=29)):
        _tend = str((end_date.date() - timedelta(days=29)))
        get_latest = True
    if start_date.date() > (_today - timedelta(days=29)):
        get_historic = False

    assert get_historic or get_latest, "Check start and end dates."

    if get_latest:
        latest = f"{tilde_url}/{domain}/{station}/{name}/{sensor}/{method}/{aspect}/latest/30d"
        rval = requests.get(latest, headers={"Accept": "text/csv"})
        if rval.status_code != 200:
            msg = f"Download of {name} for {station} failed with status code {rval.status_code}"
            msg += f" and url {latest}"
            raise ValueError(msg)
        df_latest = pd.read_csv(
            io.StringIO(rval.text),
            index_col="timestamp",
            parse_dates=["timestamp"],
            usecols=["timestamp", "value", "error"],
            date_format="ISO8601",
        )
    if get_historic:
        historic = f"{tilde_url}/{domain}/{station}/{name}/{sensor}/{method}/{aspect}/"
        historic += f"{_tstart}/{_tend}"
        rval = requests.get(historic, headers={"Accept": "text/csv"})
        if rval.status_code != 200:
            msg = f"Download of {name} for {station} failed with status code {rval.status_code}"
            msg += f" and url {historic}"
            raise ValueError(msg)
        data = io.StringIO(rval.text)
        df_historic = pd.read_csv(
            data,
            index_col="timestamp",
            parse_dates=["timestamp"],
            usecols=["timestamp", "value", "error"],
            date_format="ISO8601",
        )
        if get_latest and len(df_latest) > 0:
            df = df_historic.combine_first(df_latest)
        else:
            df = df_historic
    else:
        df = df_latest
    df.rename(columns={"value": "obs", "error": "err"}, inplace=True)
    df.index.name = "dt"
    return df
