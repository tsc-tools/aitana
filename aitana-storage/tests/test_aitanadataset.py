import pytest
from datetime import datetime
import numpy as np

from aitana.storage import Storage, generate_test_data
from aitana.storage.storage import SliceBatchSampler

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, RandomSampler


def test_basic(tmp_path_factory):
    """Test basic functionality with single time step."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 2)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset
    dataset = g.to_pytorch(['rsam', 'dsar'])
    
    # Check dataset length
    assert len(dataset) > 0
    
    # Get first sample
    sample = dataset[0]
    
    # Check that sample is a 2D tensor
    assert torch.is_tensor(sample)
    assert sample.ndim == 2
    
    # Check shape: [window_size=1, num_features=2]
    assert sample.shape == (1, 2)
    
    # Check tensor dtype
    assert sample.dtype == torch.float32


def test_with_dataloader(tmp_path_factory):
    """Test that PyTorch dataset works with DataLoader."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 1, 12)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset and dataloader
    dataset = g.to_pytorch(['rsam'])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # Iterate through dataloader
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        # Check batch structure - should be a 3D tensor [batch_size, window_size, num_features]
        assert torch.is_tensor(batch)
        assert batch.ndim == 3
        # Check batch size (last batch might be smaller)
        assert batch.shape[0] <= 4
        # window_size=1, num_features=1
        assert batch.shape[1] == 1
        assert batch.shape[2] == 1
    
    # Check that we got some batches
    assert batch_count > 0


def test_window_size(tmp_path_factory):
    """Test with window_size > 1."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 2)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset with window_size=5
    window_size = 5
    dataset = g.to_pytorch(['rsam'], window_size=window_size)
    
    # Get first sample
    sample = dataset[0]
    
    # Check that the sample has the correct shape: [window_size, num_features]
    assert sample.shape == (window_size, 1)


def test_single_feature(tmp_path_factory):
    """Test with a single feature."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 1, 12)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset with single feature
    dataset = g.to_pytorch(['rsam'])
    
    # Check that it works
    assert len(dataset) > 0
    sample = dataset[0]
    # Shape should be [window_size=1, num_features=1]
    assert sample.shape == (1, 1)


def test_multiple_features(tmp_path_factory):
    """Test with multiple features maintains order."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 1, 12)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset with multiple features
    dataset = g.to_pytorch(['rsam', 'dsar'])
    
    # Get first sample
    sample = dataset[0]
    
    # Shape should be [window_size=1, num_features=2]
    assert sample.shape == (1, 2)
    
    # Verify the features are in the correct order by checking values
    # Get the actual data to compare
    rsam_val = g('rsam').isel(datetime=0).values
    dsar_val = g('dsar').isel(datetime=0).values
    
    # First column should be rsam, second should be dsar
    assert np.isclose(sample[0, 0].item(), rsam_val, rtol=1e-5)
    assert np.isclose(sample[0, 1].item(), dsar_val, rtol=1e-5)


def test_no_time_range(tmp_path_factory):
    """Test that raises error when starttime/endtime not set."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    
    # Try to create dataset without setting time range
    with pytest.raises(ValueError, match="starttime and endtime"):
        dataset = g.to_pytorch(['rsam'])


def test_missing_feature(tmp_path_factory):
    """Test that raises error for non-existent feature."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 1, 12)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Try to create dataset with non-existent feature
    with pytest.raises(FileNotFoundError, match="non_existent_feature"):
        dataset = g.to_pytorch(['non_existent_feature'])


def test_shuffle_dataloader(tmp_path_factory):
    """Test that PyTorch dataset works with shuffled DataLoader."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 1, 12)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create PyTorch dataset and dataloader with shuffle=True
    dataset = g.to_pytorch(['rsam', 'dsar'])
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Iterate through dataloader
    batch_count = 0
    for batch in dataloader:
        batch_count += 1
        # Check batch is a 3D tensor [batch_size, window_size, num_features]
        assert torch.is_tensor(batch)
        assert batch.ndim == 3
        assert batch.shape[1] == 1  # window_size
        assert batch.shape[2] == 2  # num_features
    
    assert batch_count > 0


def test_stride(tmp_path_factory):
    """Test dataset with stride parameter."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    enddate = datetime(2023, 1, 2)
    
    # Generate and save test data
    xdf = generate_test_data(dim=1, ndays=3, tstart=startdate)
    g.save(xdf)
    
    # Set time range
    g.starttime = startdate
    g.endtime = enddate
    
    # Create dataset with stride=2
    window_size = 5
    stride = 2
    dataset = g.to_pytorch(['rsam'], window_size=window_size, stride=stride)
    
    # Get first two samples
    sample0 = dataset[0]
    sample1 = dataset[1]
    
    # Check shapes
    assert sample0.shape == (window_size, 1)
    assert sample1.shape == (window_size, 1)
    
    # Verify that sample1 starts 'stride' positions after sample0
    # Get the actual data
    rsam_data = g('rsam')
    
    # First sample should start at index 0
    expected_sample0_start = rsam_data.isel(datetime=0).values
    assert np.isclose(sample0[0, 0].item(), expected_sample0_start, rtol=1e-5)
    
    # Second sample should start at index stride (2)
    expected_sample1_start = rsam_data.isel(datetime=stride).values
    assert np.isclose(sample1[0, 0].item(), expected_sample1_start, rtol=1e-5)
    
    # Calculate expected length with stride
    # Formula: (total_timesteps - window_size) // stride + 1
    total_timesteps = len(rsam_data.datetime)
    expected_length = (total_timesteps - window_size) // stride + 1
    assert len(dataset) == expected_length


def save_rsam_and_ssam(rootdir, startdate):
    """Save 1D rsam and 2D ssam, whose frequency rows differ, in a substore."""
    g = Storage('volcanoes', rootdir=rootdir)
    store = g.get_substore('WIZ', '00', 'HHZ')
    store.save(generate_test_data(dim=1, ndays=1, tstart=startdate, feature_names=['rsam']))
    ssam = generate_test_data(dim=2, ndays=1, tstart=startdate, nfreqs=3,
                              feature_names=['ssam'], freq_names=['frequency'])
    ssam['ssam'] = ssam['ssam'] * (1 + ssam['frequency'])
    store.save(ssam)
    g.starttime = startdate
    g.endtime = datetime(2023, 1, 1, 12)
    return g, store


def test_feature_with_frequencies(tmp_path_factory):
    """A 2D feature contributes one column per frequency, in stored order."""
    g, store = save_rsam_and_ssam(tmp_path_factory.mktemp('data'), datetime(2023, 1, 1))

    dataset = store.to_pytorch(['rsam', 'ssam'])

    assert dataset.n_features == 4
    sample = dataset[5]
    assert sample.shape == (1, 4)
    np.testing.assert_allclose(sample[0, 0].numpy(), store('rsam').isel(datetime=5), rtol=1e-6)
    np.testing.assert_allclose(sample[0, 1:].numpy(), store('ssam').isel(datetime=5), rtol=1e-6)


def test_different_resolutions(tmp_path_factory):
    """Features at different resolutions raise instead of being misaligned."""
    rootdir = tmp_path_factory.mktemp('data')
    g = Storage('volcanoes', rootdir=rootdir)
    startdate = datetime(2023, 1, 1)
    g.save(generate_test_data(dim=1, ndays=1, tstart=startdate, feature_names=['rsam']))
    g.save(generate_test_data(dim=1, ndays=1, tstart=startdate, freq='1h', feature_names=['dsar']))
    g.starttime = startdate
    g.endtime = datetime(2023, 1, 1, 12)

    with pytest.raises(ValueError, match="different time stamps"):
        g.to_pytorch(['rsam', 'dsar'])


def test_transform_and_datetimes(tmp_path_factory):
    """The transform applies to every sample, and datetimes holds the time steps."""
    g, store = save_rsam_and_ssam(tmp_path_factory.mktemp('data'), datetime(2023, 1, 1))

    plain = store.to_pytorch(['rsam', 'ssam'], window_size=3, stride=2)
    doubled = store.to_pytorch(['rsam', 'ssam'], window_size=3, stride=2,
                               transform=lambda block: 2 * block)

    np.testing.assert_allclose(doubled[4].numpy(), 2 * plain[4].numpy())
    np.testing.assert_array_equal(plain.datetimes, store('rsam').datetime.values)


def test_lazy_matches_eager(tmp_path_factory):
    """Lazy reads give the same samples as loading everything, however they are requested."""
    startdate = datetime(2023, 1, 1)
    g, store = save_rsam_and_ssam(tmp_path_factory.mktemp('data'), startdate)
    options = dict(window_size=3, stride=2, transform=lambda block: 2 * block)
    eager = store.to_pytorch(['rsam', 'ssam'], **options)
    lazy = store.to_pytorch(['rsam', 'ssam'], lazy=True, **options)

    expected = torch.stack([eager[i] for i in range(len(eager))])
    assert len(lazy) == len(eager)
    for dataset in (eager, lazy):
        sampler = SliceBatchSampler(range(len(dataset)), 4)
        by_index = torch.stack([dataset[i] for i in range(len(dataset))])
        by_slice = torch.cat(list(DataLoader(dataset, batch_sampler=sampler)))
        by_list = torch.cat(list(DataLoader(dataset, batch_size=4)))
        for actual in (by_index, by_slice, by_list):
            np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    # Reading doesn't move the caller's time range
    assert store.starttime == startdate and g.endtime == datetime(2023, 1, 1, 12)


def test_slice_batch_sampler():
    """SliceBatchSampler covers a range in consecutive slices and rejects anything else."""
    assert list(SliceBatchSampler(range(10), 4)) == [slice(0, 4), slice(4, 8), slice(8, 10)]
    assert list(SliceBatchSampler(range(5, 10), 4)) == [slice(5, 9), slice(9, 10)]
    assert list(SliceBatchSampler(range(10), 4, drop_last=True)) == [slice(0, 4), slice(4, 8)]
    assert list(SliceBatchSampler(range(0), 4)) == []
    assert len(SliceBatchSampler(range(10), 4)) == 3
    assert len(SliceBatchSampler(range(10), 4, drop_last=True)) == 2
    with pytest.raises(TypeError):
        SliceBatchSampler(RandomSampler(range(10)), 4)
    with pytest.raises(TypeError):
        SliceBatchSampler([2, 9, 5], 4)
    with pytest.raises(ValueError):
        SliceBatchSampler(range(0, 10, 2), 4)
