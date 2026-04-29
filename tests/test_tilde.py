from datetime import datetime
from aitana.tilde import tilde_request


def test_tilde_request():
    """
    Test that the returned dataframe is consistent with 
    the data returned by the tilde exploration tool (https://tilde.geonet.org.nz/ui/data-discovery).
    """
    start_date = '2019-01-01'
    end_date = '2019-01-03'
    df = tilde_request(start_date=datetime(2019, 1, 1), end_date=datetime(2019, 1, 3),
                       domain="envirosensor",
                       name="lake-temperature",
                       station="RU001", sensor="01")
    assert len(df) == 70
