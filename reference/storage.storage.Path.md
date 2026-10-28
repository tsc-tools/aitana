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
