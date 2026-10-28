## access.util.cache_dataframe()


Decorator to cache pandas DataFrames, handle date ranges, and persist the cache to disk.


Usage

``` python
access.util.cache_dataframe(cache_dir="")
```


Args: cache_key (str): The attribute name for the cached DataFrame. use_class_attrs (bool): If True, fetch start_date and end_date from the class instance if available. cache_file (str): Filepath for persisting the cache.
