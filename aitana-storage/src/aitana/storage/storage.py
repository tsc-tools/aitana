from datetime import datetime
from functools import lru_cache
import logging
import os
import threading
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

try:
    import torch
except ImportError:
    torch = None

from .ingest import IngestWorker, enqueue_dataset
from .utils import get_dt, snap_to_grid
from .xarray2netcdf import xarray2netcdf
from .xarray2zarr import xarray2zarr


logger = logging.getLogger(__name__)


def _snap_time_axis(times):
    """
    Snap the time axis of an archive to its grid.

    Archive axes are evenly spaced, but NetCDF keeps times as float hours,
    which decode up to a microsecond off the grid, and archives whose
    resolution was inferred from such times drift by a fraction of a second
    over the years since the archive start.

    Parameters
    ----------
    times : np.ndarray
        datetime64 time axis of an archive, in stored order.

    Returns
    -------
    np.ndarray
        The time axis on its grid, anchored at its first time stamp.
    """
    if len(times) < 2:
        return times
    return snap_to_grid(times, get_dt(times))


@lru_cache(maxsize=4096)
def _record_info(feature_path, mtime_ns, feature, engine):
    """
    Count the timestamps of a stored feature that hold data.

    The writers pre-allocate the time axis from `archive_starttime`, so that
    data can be written out of order, and unwritten slots read as NaN. A
    timestamp of a 2D feature holds data if any of its values is not NaN.

    Parameters
    ----------
    feature_path : str
        NetCDF file or Zarr store of the feature.
    mtime_ns : int
        Modification time of `feature_path`. Only part of the cache key, so
        that a feature is read again after every write.
    feature : str
        Feature name.
    engine : str
        xarray engine to open it with.

    Returns
    -------
    tuple of (int, str or None, str or None)
        Number of records, and the times of the first and last (ISO 8601,
        seconds); the times are None when there are no records.
    """
    with xr.open_dataset(feature_path, group='original', engine=engine) as ds:
        has_data = ds[feature].notnull()
        other_dims = [dim for dim in has_data.dims if dim != 'datetime']
        if other_dims:
            has_data = has_data.any(dim=other_dims)
        times = _snap_time_axis(has_data['datetime'].values)[has_data.values]
        dims = {dim: ds[dim].values.tolist() for dim in other_dims}
    if times.size == 0:
        return 0, None, None, dims
    return (int(times.size), np.datetime_as_string(times[0], unit='s'),
            np.datetime_as_string(times[-1], unit='s'), dims)


class Path(object):
    def __init__(self, name, parentdir, create=True, backend='zarr',
                 archive_starttime=datetime(2000, 1, 1), ingest_config=None):
        self.name = name
        self.create = create
        self.backend = backend
        self.archive_starttime = archive_starttime
        self.engine = 'h5netcdf' if self.backend == 'netcdf' else self.backend
        self.path = os.path.join(parentdir, name)
        if create:
            try:
                os.makedirs(self.path, exist_ok=True)
            except FileExistsError:
                pass
        else:
            if not os.path.exists(self.path):
                raise FileNotFoundError(f"Path {self.path} not found")
        self.children = {}
        self.ingest_config = ingest_config.copy() if ingest_config else None

    def __str__(self):
        return self.path

    def __getitem__(self, key):
        if key is None:
            raise ValueError("Key cannot be None")
        try:
            return self.children[key]
        except KeyError:
            self.children[key] = Path(
                key, self.path, self.create, self.backend, self.archive_starttime,
                ingest_config=self.ingest_config)
            return self.children[key]

    def feature_path(self, feature):

        if self.backend == 'netcdf':
            file_ending = '.nc'
        elif self.backend == 'zarr':
            file_ending = '.zarr'
        _feature_path = os.path.join(self.path, feature + file_ending)
        if not os.path.exists(_feature_path):
            raise FileNotFoundError(f"File {_feature_path} not found")
            self.children[feature] = Path(feature + file_ending, self.path)
        return _feature_path

    def __call__(self, feature, group='original', metadata=False):
        """
        Request a particular feature

        :param feature: Feature name
        :type feature: str

        """
        filename = self.feature_path(feature)

        if metadata:
            with xr.open_dataset(filename, group='meta', engine=self.engine) as ds:
                return ds

        if self.endtime < self.starttime:
            raise ValueError('Startime has to be smaller than endtime.')

        logger.debug(
            f"Reading feature {feature} between {self.starttime} and {self.endtime}")

        xd_index = dict(datetime=slice(self.starttime, self.endtime))
        with xr.open_dataset(filename, group=group, engine=self.engine) as ds:
            # Snap before selecting, so that the selection uses the grid
            ds = ds.assign_coords(datetime=_snap_time_axis(ds['datetime'].values))
            rq = ds[feature].loc[xd_index].load()
            rq.attrs = ds.attrs

        return rq

    def load(self, *args, **kwargs):
        """
        Load a feature from disk
        """
        self.__call__(*args, **kwargs)

    def save(self, data, **kwargs):
        """
        Save a feature to disk
        """
        if 'archive_starttime' in kwargs:
            raise TypeError(
                "'archive_starttime' cannot be passed to save(); set it when "
                "constructing the Storage/Path instance instead.")

        if self.ingest_config and self.ingest_config.get('queue_path'):
            enqueue_dataset(
                data,
                target_path=self.path,
                backend=self.backend,
                ingest_config=self.ingest_config,
                save_kwargs=kwargs,
            )
            logger.debug("Queued data for %s backend at %s",
                         self.backend, self.path)
            return

        if self.backend == 'netcdf':
            xarray2netcdf(data, self.path,
                          archive_starttime=self.archive_starttime, **kwargs)
        elif self.backend == 'zarr':
            xarray2zarr(data, self.path,
                        archive_starttime=self.archive_starttime, **kwargs)

    def shape(self, feature):
        """
        Get shape of a feature on disk
        """
        filename = self.feature_path(feature)
        with xr.open_dataset(filename, group='original', engine=self.engine) as ds:
            return ds[feature].sizes

    def get_feature_info(self, feature):
        """
        Get metadata about a feature without loading its data.

        :param feature: Feature name
        :type feature: str
        :return: dict with keys name, recordCount (timestamps that hold data),
            earliestRecord and latestRecord (None without records)
        :rtype: dict
        """
        filename = self.feature_path(feature)
        count, earliest, latest, dims = _record_info(
            filename, os.stat(filename).st_mtime_ns, feature, self.engine)
        info = {
            'name': feature,
            'recordCount': count,
            'earliestRecord': earliest,
            'latestRecord': latest,
        }
        if dims:
            info['dims'] = dims
        return info

    def to_pytorch(self, features, window_size=1, stride=1, transform=None,
                   lazy=False):
        """
        Create a PyTorch Dataset from the store that can be used with DataLoader.

        :param features: List of feature names to include in the dataset
        :type features: list of str
        :param window_size: Number of consecutive time steps to include in each sample (default: 1)
        :type window_size: int
        :param stride: Step size between consecutive windows (default: 1)
        :type stride: int
        :param transform: Applied to blocks of time steps before they are cut
            into samples; see :class:`AitanaDataset`
        :type transform: callable, optional
        :param lazy: Read from disk on every request instead of loading all
            data up front; see :class:`AitanaDataset`
        :type lazy: bool
        :return: PyTorch Dataset instance
        :rtype: AitanaDataset

        >>> import datetime
        >>> from torch.utils.data import DataLoader
        >>> g = Storage('Whakaari', '/tmp')
        >>> g.starttime = datetime.datetime(2012,1,1,0,0,0)
        >>> g.endtime = datetime.datetime(2012,1,2,23,59,59)
        >>> dataset = g.to_pytorch(['rsam', 'dsar'])
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        >>> for batch in dataloader:
        ...     # batch is a 3D tensor with shape [batch_size, window_size, num_features]
        ...     pass
        """
        if torch is None:
            raise ImportError(
                "PyTorch is required for to_pytorch(). "
                "Install it with: pip install aitana-storage[pytorch]")
        return AitanaDataset(self, features, window_size, stride,
                             transform=transform, lazy=lazy)


class Storage(Path):
    """
    Query computed features

    :param rootdir: Path to parent directory.
    :type rootdir: str
    :param starttime: Begin of request
    :type starttime: :class:`datetime.datetime`
    :param endtime: Begin of request
    :type endtime: :class:`datetime.datetime`

    >>> import datetime
    >>> g = Storage('Whakaari', /tmp)
    >>> start = datetime.datetime(2012,1,1,0,0,0)
    >>> end = datetime.datetime(2012,1,2,23,59,59)
    >>> g.starttime = start
    >>> g.endtime = end
    >>> c = g.channel(site='WIZ', sensor='00', channel='HHZ')
    >>> rsam = c("rsam")
    """

    def __init__(self, name, rootdir, starttime=None, endtime=None, create=True, backend='netcdf',
                 ingest_config=None, archive_starttime=datetime(2000, 1, 1)):
        self.stores = set()
        self.starttime = starttime
        self.endtime = endtime
        self.archive_starttime = archive_starttime
        self._ingest_worker: Optional[IngestWorker] = None
        self._ingest_thread: Optional[threading.Thread] = None
        self._ingest_stop_event: Optional[threading.Event] = None
        super().__init__(name, rootdir, create, backend, archive_starttime,
                         ingest_config=ingest_config)

    def print_tree(self, site, indent=0, output=''):
        output += ' ' * indent + site.path + '\n'
        for site in site.children.values():
            output += self.print_tree(site, indent + 2)
        return output

    def __repr__(self):
        rstr = f"Group: {self.name}\n"
        rstr = self.print_tree(self, 0, rstr)
        return rstr

    def get_substore(self, *args):
        # return the store for a given site, sensor, or channel
        # if one of them is None return the store for the level above
        # if all are None return the root store
        try:
            st = self
            for arg in args:
                st = st[arg]
        except KeyError:
            return self

        st.starttime = self.starttime
        st.endtime = self.endtime
        self.stores.add(st)
        return st

    def from_directory(self):
        """
        Construct the storage group from the root directory
        """
        for root, dirs, files in os.walk(self.path):
            if files:
                try:
                    subdirs = root.split(self.path)[1].split(os.sep)[1:]
                except IndexError:
                    st = self.get_substore()
                else:
                    try:
                        st = self.get_substore(*subdirs)
                    except TypeError as e:
                        raise e
                for _f in files:
                    if _f.endswith('.nc'):
                        st.feature_path(_f.replace(
                            '.nc', '').replace('.zarr', ''))

    @staticmethod
    def directory_tree_to_dict(path):
        name = os.path.basename(path)
        if name.endswith('.zarr'):
            feature_name = name.replace('.zarr', '')
            return {feature_name: feature_name}
        elif os.path.isdir(path):
            dir_contents = os.listdir(path)
            children = {}
            for child in sorted(dir_contents):
                result = Storage.directory_tree_to_dict(
                    os.path.join(path, child))
                if result is not None:
                    children.update(result)
            return {name: children}
        else:
            if name.endswith('.nc'):
                feature_name = name.replace('.nc', '')
                return {feature_name: feature_name}
            else:
                return None

    def _build_feature_dict_with_url(self, feature_path, feature_name, base_url=None):
        """Build a feature info dict with metadata and an optional pre-built request URL."""
        count, earliest, latest, dims = _record_info(
            feature_path, os.stat(feature_path).st_mtime_ns, feature_name, self.engine)
        info = {
            'name': feature_name,
            'recordCount': count,
            'earliestRecord': earliest,
            'latestRecord': latest,
        }
        if dims:
            info['dims'] = dims
        if base_url is not None:
            feature_dir = os.path.dirname(feature_path)
            rel = os.path.relpath(feature_dir, self.path)
            subdirs = rel.split(os.sep) if rel != '.' else []
            path_str = ','.join([self.name] + subdirs)
            info['url'] = None if count == 0 else (
                f"{base_url}feature?path={path_str}&name={feature_name}"
                f"&starttime={earliest}&endtime={latest}"
            )
        return info

    def _enrich_tree(self, path, base_url=None):
        """Recursively build the directory tree, replacing feature names with info dicts."""
        name = os.path.basename(path)
        if name.endswith('.zarr'):
            feature_name = name.replace('.zarr', '')
            return {feature_name: self._build_feature_dict_with_url(path, feature_name, base_url)}
        elif os.path.isdir(path):
            dir_contents = os.listdir(path)
            children = {}
            for child in sorted(dir_contents):
                result = self._enrich_tree(os.path.join(path, child), base_url)
                if result is not None:
                    children.update(result)
            return {name: children}
        else:
            if name.endswith('.nc'):
                feature_name = name.replace('.nc', '')
                return {feature_name: self._build_feature_dict_with_url(path, feature_name, base_url)}
            return None

    def to_dict_with_info(self, base_url=None):
        """
        Convert the storage group to a dict with feature metadata.

        :param base_url: Base URL of the API (e.g. 'http://localhost:8000/').
            When provided, each feature dict includes a ready-to-use ``url``
            field that points to the ``/feature`` endpoint.
        :type base_url: str, optional
        :return: Nested dict mirroring the directory tree with feature info dicts at the leaves.
        :rtype: dict
        """
        return self._enrich_tree(self.path, base_url)

    def to_dict(self):
        """
        Convert the storage group to json
        """
        return Storage.directory_tree_to_dict(self.path)

    def get_starttime(self):
        return self.__starttime

    def set_starttime(self, time):
        if time is None:
            self.__starttime = None
            self.__sdate = None
            return
        self.__starttime = time
        self.__sdate = '{}{:02d}{:02d}'.format(time.year,
                                               time.month,
                                               time.day)
        for s in self.stores:
            if s is not self:
                s.starttime = time

    def get_endtime(self):
        return self.__endtime

    def set_endtime(self, time):
        if time is None:
            self.__endtime = None
            self.__edate = None
            return
        self.__endtime = time
        self.__edate = '{}{:02d}{:02d}'.format(time.year,
                                               time.month,
                                               time.day)
        for s in self.stores:
            if s is not self:
                s.endtime = time

    starttime = property(get_starttime, set_starttime)
    endtime = property(get_endtime, set_endtime)

    def _ensure_ingest_worker(self, poll_interval=None) -> IngestWorker:
        if not (self.ingest_config and self.ingest_config.get('queue_path')):
            raise RuntimeError(
                "Ingestion queue is not configured for this Storage instance.")

        if self._ingest_worker is None:
            queue_path = self.ingest_config['queue_path']
            poll = poll_interval or self.ingest_config.get(
                'poll_interval', 10.0)
            self._ingest_worker = IngestWorker(
                queue_path=queue_path,
                poll_interval=poll
            )
        elif poll_interval:
            self._ingest_worker.poll_interval = poll_interval
        return self._ingest_worker

    def run_ingest_once(self, poll_interval=None) -> int:
        worker = self._ensure_ingest_worker(poll_interval)
        return worker.run_once()

    def start_ingest_worker(self, *, background=True, poll_interval=None):
        worker = self._ensure_ingest_worker(poll_interval)
        if not background:
            return worker.run_once()
        if self._ingest_thread and self._ingest_thread.is_alive():
            return self._ingest_thread
        stop_event = threading.Event()
        thread = threading.Thread(
            target=worker.run_forever,
            kwargs={'stop_event': stop_event},
            daemon=True,
            name=f"aitana-ingest-{self.name}",
        )
        thread.start()
        self._ingest_thread = thread
        self._ingest_stop_event = stop_event
        return thread

    def stop_ingest_worker(self, timeout=None):
        if self._ingest_thread and self._ingest_thread.is_alive():
            if self._ingest_stop_event:
                self._ingest_stop_event.set()
            self._ingest_thread.join(timeout=timeout)
        self._ingest_thread = None
        self._ingest_stop_event = None


class AitanaDataset:
    """
    PyTorch Dataset of features from a store.

    Each sample is a float32 tensor of shape ``[window_size, n_features]``.
    Features are combined in the order given. A feature with dimensions
    besides ``datetime`` (e.g. the frequencies of SSAM) contributes one column
    per value of those dimensions, in stored order. Sample ``i`` covers
    ``datetimes[i * stride : i * stride + window_size]``.

    All features must have the same time stamps between the store's
    ``starttime`` and ``endtime`` (both inclusive); features at different
    resolutions raise a ValueError instead of being combined. The time range
    is read once, so later changes to the store's times don't affect the
    dataset.

    Parameters
    ----------
    storage : Path
        Store or substore holding the features, with ``starttime`` and
        ``endtime`` set.
    features : str or list of str
        Feature names.
    window_size : int, optional
        Number of consecutive time steps per sample.
    stride : int, optional
        Time steps between the starts of consecutive samples.
    transform : callable, optional
        Applied to blocks of consecutive time steps, float32 arrays of shape
        ``[n_timesteps, n_features]``, before they are cut into samples. It
        must return an array of the same shape and treat each time step on its
        own (e.g. scale it), since the length of the blocks varies.
    lazy : bool, optional
        Read from disk on every request instead of loading all data when the
        dataset is created. Iterate a lazy dataset with `SliceBatchSampler`,
        which makes each batch one read per feature; any other access reads
        one sample at a time.

    Attributes
    ----------
    n_features : int
        Number of columns of each sample.
    datetimes : np.ndarray
        Time stamps of all time steps in the dataset.
    """

    def __init__(self, storage, features, window_size=1, stride=1,
                 transform=None, lazy=False):
        starttime = getattr(storage, 'starttime', None)
        endtime = getattr(storage, 'endtime', None)
        if starttime is None or endtime is None:
            raise ValueError("Storage must have starttime and endtime set")

        self.storage = storage
        self.features = features if isinstance(features, list) else [features]
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.lazy = lazy

        # File and position of the first time step in the time range, per feature
        self._sources = []
        times = {}
        self.n_features = 0
        for feature in self.features:
            try:
                path = storage.feature_path(feature)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Feature '{feature}' not found in storage")
            with xr.open_dataset(path, group='original', engine=storage.engine) as ds:
                index = pd.DatetimeIndex(_snap_time_axis(ds['datetime'].values))
                first = index.searchsorted(starttime, side='left')
                last = index.searchsorted(endtime, side='right')
                times[feature] = index[first:last].values
                sizes = ds[feature].sizes
            self.n_features += int(np.prod(
                [size for dim, size in sizes.items() if dim != 'datetime']))
            self._sources.append((feature, path, first))

        self.datetimes = times[self.features[0]]
        for feature in self.features[1:]:
            if not np.array_equal(times[feature], self.datetimes):
                raise ValueError(
                    f"Features '{self.features[0]}' and '{feature}' have different "
                    f"time stamps between {starttime} and {endtime} "
                    f"({len(self.datetimes)} and {len(times[feature])} time steps), "
                    "e.g. because their resolutions differ, so they cannot be "
                    "combined in one dataset.")

        total_timesteps = len(self.datetimes)
        # Number of windows = (total_timesteps - window_size) // stride + 1
        if total_timesteps < window_size:
            self.length = 0
        else:
            self.length = (total_timesteps - window_size) // stride + 1

        if self.length <= 0:
            raise ValueError(
                f"Not enough data points for window_size={window_size} with stride={stride}. "
                f"Available data points: {total_timesteps}"
            )

        if not lazy:
            self._block = self._read(0, total_timesteps)

    def _read(self, start, stop):
        """
        Read a block of consecutive time steps of all features.

        Parameters
        ----------
        start, stop : int
            Positions in `datetimes` of the first and one past the last time
            step.

        Returns
        -------
        np.ndarray
            float32 array of shape ``[stop - start, n_features]``, with
            `transform` applied.
        """
        columns = []
        for feature, path, first in self._sources:
            with xr.open_dataset(path, group='original', engine=self.storage.engine) as ds:
                values = ds[feature].isel(datetime=slice(first + start, first + stop))
                values = values.transpose('datetime', ...).values
            columns.append(values.reshape(len(values), -1))
        block = np.concatenate(columns, axis=1).astype(np.float32)
        if self.transform is not None:
            block = np.asarray(self.transform(block), dtype=np.float32)
        return block

    def _windows(self, block, starts):
        """
        Cut samples from a block of time steps.

        Parameters
        ----------
        block : np.ndarray
            Array of shape ``[n_timesteps, n_features]``.
        starts : np.ndarray
            Position in `block` of the first time step of each sample.

        Returns
        -------
        np.ndarray
            Array of shape ``[len(starts), window_size, n_features]``.
        """
        return block[starts[:, np.newaxis] + np.arange(self.window_size)]

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.length

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``[window_size, n_features]``.
        """
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}")
        start = idx * self.stride
        if self.lazy:
            window = self._read(start, start + self.window_size)
        else:
            window = self._block[start:start + self.window_size]
        return torch.from_numpy(np.array(window))

    def __getitems__(self, indices):
        """
        Get the samples of a batch; DataLoader calls this for every batch.

        Parameters
        ----------
        indices : slice or sequence of int
            Indices of the samples, e.g. from `SliceBatchSampler`. In lazy
            mode, a slice of consecutive indices is read in one go per feature.

        Returns
        -------
        list of torch.Tensor
            One tensor of shape ``[window_size, n_features]`` per sample.
        """
        if isinstance(indices, slice):
            indices = range(len(self))[indices]
        else:
            for idx in indices:
                if not 0 <= idx < len(self):
                    raise IndexError(
                        f"Index {idx} out of range for dataset of length {len(self)}")
        if len(indices) == 0:
            return []

        starts = np.asarray(indices) * self.stride
        if not self.lazy:
            windows = self._windows(self._block, starts)
        elif isinstance(indices, range) and indices.step == 1:
            block = self._read(starts[0], starts[-1] + self.window_size)
            windows = self._windows(block, starts - starts[0])
        else:
            windows = np.stack(
                [self._read(start, start + self.window_size) for start in starts])
        return list(torch.from_numpy(windows))


class SliceBatchSampler:
    """
    Batch sampler that yields slices of consecutive sample indices.

    With `AitanaDataset`, each batch is then one contiguous read, which is
    what makes a lazy dataset fast to iterate. Batches come in order.

    Parameters
    ----------
    indices : range
        Consecutive sample indices to cover, e.g. ``range(len(dataset))``.
    batch_size : int
        Number of samples per batch.
    drop_last : bool, optional
        Drop the last batch if it is smaller than `batch_size`.

    Examples
    --------
    >>> list(SliceBatchSampler(range(10), 4))
    [slice(0, 4, None), slice(4, 8, None), slice(8, 10, None)]
    >>> loader = DataLoader(dataset, batch_sampler=SliceBatchSampler(range(len(dataset)), 64))
    """

    def __init__(self, indices, batch_size, drop_last=False):
        if not isinstance(indices, range):
            raise TypeError(
                "SliceBatchSampler needs a range of indices, e.g. "
                f"range(len(dataset)), not {type(indices).__name__}.")
        if indices.step != 1:
            raise ValueError(
                "SliceBatchSampler needs consecutive indices, a range with step 1.")
        self.indices = indices
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        stop = self.indices.stop
        for start in range(self.indices.start, stop, self.batch_size):
            if self.drop_last and start + self.batch_size > stop:
                return
            yield slice(start, min(start + self.batch_size, stop))

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        return (len(self.indices) + self.batch_size - 1) // self.batch_size
