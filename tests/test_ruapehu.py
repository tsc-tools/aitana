import os
from datetime import datetime
import pytest
from aitana.ruapehu import CraterLake, Gas, Seismicity, eruptions


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


def test_seismicity():
    tstart = datetime(2023, 12, 1)
    tend = datetime(2023, 12, 31)
    g = Seismicity(tstart, tend)
    cat1 = g.regional()
    cat2 = g.cone()
    cat3 = g.rm_duplicates(cat1, cat2)
    assert cat1.shape == (41, 22)
    assert cat2.shape == (10, 22)
    assert cat3.shape == (31, 22)


def test_eruptions():
    e1 = eruptions(min_size=3)
    e2 = eruptions(min_size=3, dec_interval='14D')
    assert e1.shape == (293, 5)
    assert e2.shape == (57, 5)


def test_rsam():
    tstart = datetime(2007, 1, 1)
    tend = datetime(2024, 11, 30)
    g = Seismicity(tstart, tend)
    rsam = g.daily_rsam()
    assert rsam.shape == (6158, 1)
