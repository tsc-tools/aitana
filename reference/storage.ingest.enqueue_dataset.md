## storage.ingest.enqueue_dataset()


Enqueue a dataset for ingestion.


Usage

``` python
storage.ingest.enqueue_dataset(
    data, target_path, *, backend, ingest_config, save_kwargs=None
)
```


## Parameters


`data: xr.Dataset`  
The dataset to enqueue.

`target_path: str`  
The target path where the dataset should be saved.

`backend: str`  
The backend to use for saving the dataset ('zarr' or 'netcdf').

`ingest_config: dict`  
Configuration for the ingest queue, must include 'queue_path'.

`save_kwargs: Optional[dict] = None`  
Additional keyword arguments to pass to the save function, by default None.


## Returns


`dict`  
A message dictionary representing the enqueued dataset.
