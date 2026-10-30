## storage.storage.Path


Usage

``` python
storage.storage.Path(
    name,
    parentdir,
    create=True,
    backend="zarr",
    archive_starttime=datetime(2000, 1, 1),
    ingest_config=None
)
```


## Methods

| Name | Description |
|----|----|
| [__call__()](#__call__) | Request a particular feature |
| [get_feature_info()](#get_feature_info) | Get metadata about a feature without loading its data. |
| [load()](#load) | Load a feature from disk |
| [save()](#save) | Save a feature to disk |
| [shape()](#shape) | Get shape of a feature on disk |
| [to_pytorch()](#to_pytorch) | Create a PyTorch Dataset from the store that can be used with DataLoader. |

------------------------------------------------------------------------


#### \_\_call\_\_()


Request a particular feature


Usage

``` python
__call__(feature, group="original", metadata=False)
```


## Parameters

- **feature** : `str` -- Feature name


------------------------------------------------------------------------


#### get_feature_info()


Get metadata about a feature without loading its data.


Usage

``` python
get_feature_info(feature)
```


## Parameters

- **feature** : `str` -- Feature name

## Returns

`dict`  
dict with keys name, recordCount (timestamps that hold data), earliestRecord and latestRecord (None without records)


------------------------------------------------------------------------


#### load()


Load a feature from disk


Usage

``` python
load(*args, **kwargs)
```


------------------------------------------------------------------------


#### save()


Save a feature to disk


Usage

``` python
save(data, **kwargs)
```


------------------------------------------------------------------------


#### shape()


Get shape of a feature on disk


Usage

``` python
shape(feature)
```


------------------------------------------------------------------------


#### to_pytorch()


Create a PyTorch Dataset from the store that can be used with DataLoader.


Usage

``` python
to_pytorch(features, window_size=1, stride=1, transform=None, lazy=False)
```


## Parameters

- **features** : `list of str` -- List of feature names to include in the dataset
- **window_size** : `int` -- Number of consecutive time steps to include in each sample (default: 1)
- **stride** : `int` -- Step size between consecutive windows (default: 1)
- **transform** : `callable, optional` -- Applied to blocks of time steps before they are cut into samples; see [AitanaDataset](storage.storage.AitanaDataset.md#aitana.storage.storage.AitanaDataset)
- **lazy** : `bool` -- Read from disk on every request instead of loading all data up front; see [AitanaDataset](storage.storage.AitanaDataset.md#aitana.storage.storage.AitanaDataset)

## Returns

`AitanaDataset`  
PyTorch Dataset instance

``` python
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
```
