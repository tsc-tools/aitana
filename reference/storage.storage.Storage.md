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
