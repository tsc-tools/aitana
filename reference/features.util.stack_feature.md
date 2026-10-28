## features.util.stack_feature()


Stack features in a dataarray.


Usage

``` python
features.util.stack_feature(
    xrdata,
    stack_length,
    interval="10min",
)
```


## Parameters


`xrdata: xarray.DataArray`  
The data array to stack.

`stack_length: str`  
The length of the stack.

`interval: str = ``"10min"`  
The interval between stacks, by default "10min".
