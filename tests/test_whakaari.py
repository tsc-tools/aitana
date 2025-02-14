import numpy as np
import pandas as pd
from aitana.whakaari import Gas, Seismicity, eruptions, load_all


def test_gas():
    g = Gas(start_date='2020-01-01', end_date='2024-12-31')
    df1 = g.so2(fuse=False, smooth=False)
    df2 = g.so2(fuse=True, smooth=True)
    assert (df1.obs.max() / 86.4 - 31.9) < 0.001
    assert (df1.obs.min() / 86.4 - 0.355) < 0.001
    # test that df2 is smoother than df1 by comparing standard deviations
    assert df2.obs.std() < df1.obs.std()
    df = g.co2()
    assert df.obs.loc['2023-11-09'].values[0] / 86.4 - 5.2 < 0.001

    df = g.h2s()
    assert df.obs.loc['2024-02-29'].values[0] / 86.4 - 0.29 < 0.001


def test_seismicity():
    s = Seismicity('2019-12-01', '2020-01-31')
    df = s.rsam()
    assert df.obs.max() > 7900
    dfq = s.quakes()
    assert dfq.index[0] >= pd.to_datetime('2019-12-01', utc=True)
    assert dfq.index[-1] <= pd.to_datetime('2020-01-31T23:59:59', utc=True)
    assert dfq.magnitude.max() > 4.
    assert dfq.magnitude.min() < 2.


def test_eruptions():
    df = eruptions(end_date='2024-12-31')
    assert np.diff(df.index).min().days == 0
    df = eruptions(dec_interval='10D', end_date='2024-12-31')
    assert np.diff(df.index).min().days == 10
    df = eruptions(start_date='2010-01-01', end_date='2024-12-31', min_size=2)
    assert df.shape[0] == 4


def test_load_all():
    end_date = '2025-02-14'
    df = load_all(end_date=end_date)
    assert df.iloc[-1].isnull().sum() == 0
