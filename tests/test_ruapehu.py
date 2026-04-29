from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

from aitana.ruapehu import (CraterLake,
                            Gas,
                            Seismicity,
                            eruptions,
                            load_all,
                            lake_height_scaling)


def test_crater_lake():
    tstart = datetime(2023, 1, 1)
    tend = datetime(2023, 1, 2)
    rcl = CraterLake(tstart, tend)
    df = rcl.temperature()
    assert df.obs.loc['2023-01-01'] - 30.908 < 0.001

    tstart = datetime(2022, 12, 21)
    tend = datetime(2022, 12, 22)
    rcl = CraterLake(tstart, tend)
    df = rcl.water_analyte('Mg')
    assert df.obs.loc['2022-12-21'] - 369.0 < 0.001

    tstart = datetime(2010, 11, 18)
    tend = datetime(2010, 11, 19)
    rcl = CraterLake(tstart, tend)
    df = rcl.water_level()
    assert df.obs.loc['2010-11-18'] - 1.241 < 0.001

    tstart = datetime(2021, 5, 8)
    tend = datetime(2021, 5, 9)
    rcl = CraterLake(tstart, tend)
    df = rcl.water_level()
    assert df.obs.loc['2021-05-08'] - 1.269 < 0.001

    tstart = datetime(2025, 7, 6)
    tend = datetime(2025, 7, 9)
    rcl = CraterLake(tstart, tend)
    df = rcl.water_level()
    intercept, slope = lake_height_scaling()
    assert (df.obs.loc['2025-07-08']/slope - intercept) - 1.381 < 0.001

    tstart = datetime(2025, 7, 1)
    tend = datetime(2025, 7, 10)
    rcl = CraterLake(tstart, tend)
    df = rcl.temperature(resample=None)
    ts = pd.Timestamp('2025-07-09 20:50:00+0000', tzinfo=timezone.utc)
    assert abs(df.loc[ts].obs - 11.3756) < 0.001
    df1 = rcl.temperature()
    assert abs(df1.obs.loc['2025-07-08'] - 11.423) < 0.01


def test_water_analyte_duplicates():
    tstart = datetime(2024, 1, 1)
    tend = datetime(2025, 9, 1)
    rcl = CraterLake(tstart, tend)
    df = rcl.water_analyte('Mg')
    assert df.index.duplicated().sum() == 0


def test_gas():
    tstart = datetime(2023, 12, 20)
    tend = datetime(2023, 12, 21)
    g = Gas(tstart, tend)
    df = g.so2(clear_cache=True)
    assert df.obs.loc['2023-12-20T21:30:00'] - 132.192 < 0.001

    df = g.co2(clear_cache=True)
    assert df.obs.loc['2023-12-20T21:30:00'] - 851.904 < 0.001

    df = g.h2s(clear_cache=True)
    assert df.obs.loc['2023-12-20T21:30:00'] == 0.

    tstart = datetime(2021, 1, 1)
    tend = datetime(2021, 12, 31)
    g = Gas(tstart, tend)
    df = g.so2()
    assert df.shape == (5, 2)

    with pytest.raises(ValueError):
        Gas(datetime(2021, 10, 1), datetime(2021, 10, 3)).so2()


def test_seismicity():
    tstart = datetime(2023, 12, 1)
    tend = datetime(2023, 12, 31)
    g = Seismicity(tstart, tend)
    cat1 = g.regional()
    cat2 = g.cone()
    cat3 = g.rm_duplicates(cat1, cat2)
    assert cat1.shape == (43, 20)
    assert cat2.shape == (10, 20)
    assert cat3.shape == (33, 20)

    tstart = datetime(1955, 1, 1)
    tend = datetime(2021, 6, 1)
    g1 = Seismicity(tstart, tend)
    cat1 = g1.cone()
    assert cat1.shape[0] == 10297
    cat2 = g1.regional()
    cat3 = g.rm_duplicates(cat2, cat1)
    assert cat3.shape[0] == 11501


def test_eruptions():
    e1 = eruptions(min_size=3,
                   end_date=datetime(2024, 11, 30, tzinfo=timezone.utc))
    e2 = eruptions(min_size=3, dec_interval='14D',
                   end_date=datetime(2024, 11, 30, tzinfo=timezone.utc))
    assert e1.shape == (293, 5)
    assert e2.shape == (57, 5)


def test_rsam():
    tstart = datetime(2008, 1, 1)
    tend = datetime(2024, 11, 30)
    g = Seismicity(tstart, tend)
    rsam = g.daily_rsam()
    assert rsam.shape[0] - 6159 < 21

    tstart = datetime(2023, 11, 1, tzinfo=timezone.utc)
    tend = datetime(2023, 11, 30, tzinfo=timezone.utc)
    g = Seismicity(tstart, tend)
    rsam = g.daily_rsam()
    assert rsam.shape == (30, 1)

    # # test local rsam server
    # tend = datetime(2025, 11, 26, tzinfo=timezone.utc)
    # tstart = tend - timedelta(days=30)
    # g = Seismicity(tstart, tend)
    # rsam = g.daily_rsam(update=True)
    # assert rsam.index[-1] == pd.Timestamp('2025-11-26', tz='UTC')
    # assert rsam['obs'][-1] - 31.77 < 0.01


def test_load_all():
    end_date = datetime(2024, 11, 30)
    df = load_all(end_date=end_date)
    assert df.iloc[-1].isnull().sum() == 0


def test_lake_height_scaling():
    intercept, slope = lake_height_scaling()
    assert abs(intercept - 0.274) < 0.001
    assert abs(slope - 1.207) < 0.001
