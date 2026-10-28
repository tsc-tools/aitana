## storage.storage.AitanaDataset


PyTorch Dataset wrapper for Aitana Storage.


Usage

``` python
storage.storage.AitanaDataset(
    storage,
    features,
    window_size=1,
    stride=1,
)
```


This dataset loads time series data from Storage and returns it as PyTorch tensors. Each sample is a 2D tensor with shape \[window_size, num_features\] where features are ordered as provided to **init**.


## Methods

| Name | Description |
|----|----|
| [__getitem__()](#__getitem__) | Get a sample from the dataset. |
| [__init__()](#__init__) | Initialize the dataset. |
| [__len__()](#__len__) | Return the number of samples in the dataset. |

------------------------------------------------------------------------


#### \_\_getitem\_\_()


Get a sample from the dataset.


Usage

``` python
__getitem__(idx)
```


## Parameters

- **idx** -- Index of the sample

## Returns

  
2D PyTorch tensor with shape \[window_size, num_features\] where features are in the order provided to **init**


------------------------------------------------------------------------


#### \_\_init\_\_()


Initialize the dataset.


Usage

``` python
__init__(storage, features, window_size=1, stride=1)
```


## Parameters

- **storage** -- Storage instance with starttime and endtime set
- **features** -- List of feature names to load
- **window_size** -- Number of consecutive time steps per sample
- **stride** -- Step size between consecutive windows (default: 1)


------------------------------------------------------------------------


#### \_\_len\_\_()


Return the number of samples in the dataset.


Usage

``` python
__len__()
```
