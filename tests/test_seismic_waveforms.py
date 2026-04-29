from datetime import datetime
from obspy import UTCDateTime
import pytest
import torch
from torch.utils.data import DataLoader

from aitana.seismic_waveforms import SeismicWaveforms, MockSDSWaveforms


def test_client_type():
    class TestClientType:
        pass

    client = TestClientType()
    with pytest.raises(ValueError):
        SeismicWaveforms("NZ.WIZ.10.HHZ", [client],
                         start=UTCDateTime(2020, 1, 1),
                         end=UTCDateTime(2020, 1, 2),
                         window_size=600, stride=300)


def test_len(tmp_path_factory):
    sds_dir = tmp_path_factory.mktemp("sds", numbered=True)
    client = MockSDSWaveforms(sds_dir)
    sw = SeismicWaveforms("NZ.WIZ.10.HHZ", [client],
                          start=UTCDateTime(2020, 1, 1, 12, 23, 22),
                          end=datetime(2020, 1, 2),
                          window_size=600, stride=300)
    assert len(sw) == 139


def test_getitem(tmp_path_factory):
    sds_dir = tmp_path_factory.mktemp("sds", numbered=True)
    client = MockSDSWaveforms(sds_dir)
    sw = SeismicWaveforms("NZ.WIZ.10.HHZ", [client],
                          start=UTCDateTime(2020, 1, 1, 12, 23, 22),
                          end=UTCDateTime(2020, 1, 2),
                          window_size=600, stride=300)
    arr = sw[10]
    assert arr.shape == (2, 10*600)


def test_dataloader(tmp_path_factory):
    sds_dir = tmp_path_factory.mktemp("sds", numbered=True)
    client = MockSDSWaveforms(sds_dir)
    sw = SeismicWaveforms("NZ.WIZ.10.HHZ", [client],
                          start=UTCDateTime(2020, 1, 1, 12, 23, 22),
                          end=UTCDateTime(2020, 1, 2),
                          window_size=600, stride=300)
    dataloader = DataLoader(sw, batch_size=4)
    for batch in dataloader:
        assert batch.shape == (4, 2, 10*600)
        break


def test_three_components(tmp_path_factory):
    sds_dir = tmp_path_factory.mktemp("sds", numbered=True)
    client = MockSDSWaveforms(sds_dir)
    sw = SeismicWaveforms("NZ.WIZ.10.HH?", [client],
                          start=UTCDateTime(2020, 1, 1, 12, 23, 22),
                          end=UTCDateTime(2020, 1, 2),
                          window_size=600, stride=300)
    arr = sw[10]
    assert arr.shape == (6, 10*600)
