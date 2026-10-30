## storage.storage.SliceBatchSampler


Batch sampler that yields slices of consecutive sample indices.


Usage

``` python
storage.storage.SliceBatchSampler(
    indices,
    batch_size,
    drop_last=False,
)
```


With [AitanaDataset](storage.storage.AitanaDataset.md#aitana.storage.storage.AitanaDataset), each batch is then one contiguous read, which is what makes a lazy dataset fast to iterate. Batches come in order.


## Parameters


`indices: range`  
Consecutive sample indices to cover, e.g. `range(len(dataset))`.

`batch_size: int`  
Number of samples per batch.

`drop_last: bool = ``False`  
Drop the last batch if it is smaller than `batch_size`.


## Examples

``` python
>>> list(SliceBatchSampler(range(10), 4))
```

\[slice(0, 4, None), slice(4, 8, None), slice(8, 10, None)\]

``` python
>>> loader = DataLoader(dataset, batch_sampler=SliceBatchSampler(range(len(dataset)), 64))
```
