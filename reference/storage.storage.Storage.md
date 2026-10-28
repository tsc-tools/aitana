## storage.storage.Storage


Query computed features


Usage

``` python
storage.storage.Storage(
    name,
    rootdir,
    starttime=None,
    endtime=None,
    create=True,
    backend="netcdf",
    ingest_config=None,
    archive_starttime=datetime(2000, 1, 1)
)
```


## Parameters

- **rootdir** : `str` -- Path to parent directory.
- **starttime** : `datetime.datetime` -- Begin of request
- **endtime** : `datetime.datetime` -- Begin of request

``` python
>>> import datetime
>>> g = Storage('Whakaari', /tmp)
>>> start = datetime.datetime(2012,1,1,0,0,0)
>>> end = datetime.datetime(2012,1,2,23,59,59)
>>> g.starttime = start
>>> g.endtime = end
>>> c = g.channel(site='WIZ', sensor='00', channel='HHZ')
>>> rsam = c("rsam")
```


## Methods

| Name | Description |
|----|----|
| [from_directory()](#from_directory) | Construct the storage group from the root directory |
| [to_dict()](#to_dict) | Convert the storage group to json |
| [to_dict_with_info()](#to_dict_with_info) | Convert the storage group to a dict with feature metadata. |
| [to_pytorch()](#to_pytorch) | Create a PyTorch Dataset from the Storage that can be used with DataLoader. |

------------------------------------------------------------------------


#### from_directory()


Construct the storage group from the root directory


Usage

``` python
from_directory()
```


------------------------------------------------------------------------


#### to_dict()


Convert the storage group to json


Usage

``` python
to_dict()
```


------------------------------------------------------------------------


#### to_dict_with_info()


Convert the storage group to a dict with feature metadata.


Usage

``` python
to_dict_with_info(base_url=None)
```


## Parameters

- **base_url** : `str, optional` -- Base URL of the API (e.g. 'http://localhost:8000/'). When provided, each feature dict includes a ready-to-use `url` field that points to the `/feature` endpoint.

## Returns

`dict`  
Nested dict mirroring the directory tree with feature info dicts at the leaves.


------------------------------------------------------------------------


#### to_pytorch()


Create a PyTorch Dataset from the Storage that can be used with DataLoader.


Usage

``` python
to_pytorch(features, window_size=1, stride=1)
```


## Parameters

- **features** : `list of str` -- List of feature names to include in the dataset
- **window_size** : `int` -- Number of consecutive time steps to include in each sample (default: 1)
- **stride** : `int` -- Step size between consecutive windows (default: 1)

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
...     # batch is a 2D tensor with shape [batch_size, num_features]
...     pass
```
