from datetime import datetime
import inspect

import pandas as pd
import numpy as np

from aitana.util import cache_dataframe, eqRate


class DataFetcher:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    @cache_dataframe()
    def fetch_data(self):
        # Simulate fetching data for the specified date range
        date_range = pd.date_range(
            start=pd.to_datetime('2024-01-01', utc=True), end=self.end_date, freq="D")
        data = pd.DataFrame({
            "value": np.random.rand(len(date_range))
        }, index=date_range)
        return data


@cache_dataframe()
def fetch_data_function(val, end_date):
    date_range = pd.date_range(
        start=pd.to_datetime("2024-01-01", utc=True), end=end_date, freq="D")
    data = pd.DataFrame(
        {"value": np.random.rand(len(date_range)) * val}, index=date_range)
    return data


def test_cache():
    # Usage
    fetcher = DataFetcher(start_date="2024-01-01", end_date="2024-01-10")

    # Fetch using instance attributes
    df1 = fetcher.fetch_data(clear_cache=True)
    assert len(df1) == 10

    # Update instance attributes
    fetcher.start_date = "2024-01-10"
    fetcher.end_date = "2024-01-15"

    # Fetch using updated attributes
    df2 = fetcher.fetch_data()
    assert len(df2) == 6

    # Get cached DataFrame
    members = inspect.getmembers(fetcher.fetch_data)
    cached_df = [value for name,
                 value in members if name.startswith('cached_df_')][0]
    # Cached data is updated
    assert len(cached_df) == 15

    # Fetch using function arguments
    df3 = fetch_data_function(5, start_date="2024-02-01",
                              end_date="2024-02-10", clear_cache=True)
    assert len(df3) == 10

    # Update function arguments
    df4 = fetch_data_function(
        4, start_date="2024-02-10", end_date="2024-02-15")
    assert len(df4) == 6

    # Get cached DataFrame
    members = inspect.getmembers(fetch_data_function)
    cached_df = [value for name,
                 value in members if name.startswith('cached_df_')][0]

    # Cached data is updated
    assert len(cached_df) == 46


def test_eqRate():
    """
    Test earthquake rate with a synthetic example.
    """
    cat = pd.DataFrame({'publicid': np.random.rand(5)},
                       index=pd.date_range('1978-07-18',
                                           periods=5,
                                           freq='2D'))
    eqr1 = eqRate(cat, fixed_time=2, enddate=datetime(1978, 7, 27))
    eqr2 = eqRate(cat, fixed_nevents=2, enddate=datetime(1978, 7, 27))
    np.testing.assert_array_equal(eqr1.values.squeeze(),
                                  np.array([0.5]*6))
    np.testing.assert_array_equal(eqr2.values.squeeze(),
                                  np.array([0.5]*3))
