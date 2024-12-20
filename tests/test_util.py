import pandas as pd
import numpy as np

from aitana.util import cache_dataframe


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

    # Cached data is updated
    assert len(fetcher.fetch_data.cached_df) == 15

    # Fetch using function arguments
    df3 = fetch_data_function(5, start_date="2024-02-01",
                              end_date="2024-02-10", clear_cache=True)
    assert len(df3) == 10

    # Update function arguments
    df4 = fetch_data_function(
        4, start_date="2024-02-10", end_date="2024-02-15")
    assert len(df4) == 6

    # Cached data is updated
    assert len(fetch_data_function.cached_df) == 46
