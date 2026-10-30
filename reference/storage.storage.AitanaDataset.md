## storage.storage.AitanaDataset


PyTorch Dataset of features from a store.


Usage

``` python
storage.storage.AitanaDataset(
    storage, features, window_size=1, stride=1, transform=None, lazy=False
)
```


Each sample is a float32 tensor of shape `[window_size, n_features]`. Features are combined in the order given. A feature with dimensions besides `datetime` (e.g. the frequencies of SSAM) contributes one column per value of those dimensions, in stored order. Sample `i` covers `datetimes[i * stride : i * stride + window_size]`.

All features must have the same time stamps between the store's `starttime` and `endtime` (both inclusive); features at different resolutions raise a ValueError instead of being combined. The time range is read once, so later changes to the store's times don't affect the dataset.


## Parameters


`storage: Path`  
Store or substore holding the features, with `starttime` and `endtime` set.

`features: str or list of str`  
Feature names.

`window_size: int = ``1`  
Number of consecutive time steps per sample.

`stride: int = ``1`  
Time steps between the starts of consecutive samples.

`transform: callable = None`  
Applied to blocks of consecutive time steps, float32 arrays of shape `[n_timesteps, n_features]`, before they are cut into samples. It must return an array of the same shape and treat each time step on its own (e.g. scale it), since the length of the blocks varies.

`lazy: bool = ``False`  
Read from disk on every request instead of loading all data when the dataset is created. Iterate a lazy dataset with [SliceBatchSampler](storage.storage.SliceBatchSampler.md#aitana.storage.storage.SliceBatchSampler), which makes each batch one read per feature; any other access reads one sample at a time.


## Attributes


`n_features: int`  
Number of columns of each sample.

`datetimes: np.ndarray`  
Time stamps of all time steps in the dataset.


## Methods

| Name | Description |
|----|----|
| [__getitem__()](#__getitem__) | Get a sample from the dataset. |
| [__getitems__()](#__getitems__) | Get the samples of a batch; DataLoader calls this for every batch. |
| [__len__()](#__len__) | Return the number of samples in the dataset. |

------------------------------------------------------------------------


#### \_\_getitem\_\_()


Get a sample from the dataset.


Usage

``` python
__getitem__(idx)
```


##### Parameters


`idx: int`  
Index of the sample.


##### Returns


`torch.Tensor`  
Tensor of shape `[window_size, n_features]`.


------------------------------------------------------------------------


#### \_\_getitems\_\_()


Get the samples of a batch; DataLoader calls this for every batch.


Usage

``` python
__getitems__(indices)
```


##### Parameters


`indices: slice or sequence of int`  
Indices of the samples, e.g. from [SliceBatchSampler](storage.storage.SliceBatchSampler.md#aitana.storage.storage.SliceBatchSampler). In lazy mode, a slice of consecutive indices is read in one go per feature.


##### Returns


`list of torch.Tensor`  
One tensor of shape `[window_size, n_features]` per sample.


------------------------------------------------------------------------


#### \_\_len\_\_()


Return the number of samples in the dataset.


Usage

``` python
__len__()
```
