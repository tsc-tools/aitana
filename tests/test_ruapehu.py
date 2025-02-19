from datetime import datetime, timezone, timedelta

import pytest

from aitana.ruapehu import CraterLake, Gas, Seismicity, eruptions, load_all


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
    assert df.obs.loc['2022-12-21'] - 362.0 < 0.001

    tstart = datetime(2010, 11, 18)
    tend = datetime(2010, 11, 19)
    rcl = CraterLake(tstart, tend)
    df = rcl.water_level()
    assert df.obs.loc['2010-11-18'].mean() - 1.241 < 0.001

    tstart = datetime(2021, 5, 8)
    tend = datetime(2021, 5, 9)
    rcl = CraterLake(tstart, tend)
    df = rcl.water_level()
    assert df.obs.loc['2021-05-08'].mean() - 1.269 < 0.001


def test_gas():
    tstart = datetime(2023, 12, 20)
    tend = datetime(2023, 12, 21)
    g = Gas(tstart, tend)
    df = g.so2()
    assert df.obs.loc['2023-12-20T21:30:00'] - 132.192 < 0.001

    df = g.co2()
    assert df.obs.loc['2023-12-20T21:30:00'] - 851.904 < 0.001

    df = g.h2s()
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


def test_eruptions():
    e1 = eruptions(min_size=3,
                   end_date=datetime(2024, 11, 30, tzinfo=timezone.utc))
    e2 = eruptions(min_size=3, dec_interval='14D',
                   end_date=datetime(2024, 11, 30, tzinfo=timezone.utc))
    assert e1.shape == (293, 5)
    assert e2.shape == (57, 5)


def test_rsam():
    tstart = datetime(2007, 1, 1)
    tend = datetime(2024, 11, 30)
    g = Seismicity(tstart, tend)
    rsam = g.daily_rsam()
    assert rsam.shape == (6159, 1)

    tstart = datetime(2023, 11, 1, tzinfo=timezone.utc)
    tend = datetime(2023, 11, 30, tzinfo=timezone.utc)
    g = Seismicity(tstart, tend)
    rsam = g.daily_rsam()
    assert rsam.shape == (30, 1)

    # test local rsam server
    # tstart = datetime(2023, 11, 1, tzinfo=timezone.utc)
    # tend = datetime.now(timezone.utc)
    # tstart = tend - timedelta(days=30)
    # g = Seismicity(tstart, tend)
    # rsam = g.daily_rsam(update=True)
    # print(rsam)


def test_load_all():
    end_date = datetime(2024, 11, 30)
    df = load_all(end_date=end_date)
    assert df.iloc[-1].isnull().sum() == 0
