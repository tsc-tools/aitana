## volcanoes.whakaari.Gas


Usage

``` python
volcanoes.whakaari.Gas(
    start_date,
    end_date,
)
```


## Methods

| Name | Description |
|----|----|
| [so2()](#so2) | Retrieve SO2 gas flux for Whakaari. If fuse is True, the returned values |

------------------------------------------------------------------------


#### so2()


Retrieve SO2 gas flux for Whakaari. If fuse is True, the returned values


Usage

``` python
so2(fuse=False, smooth=True)
```


will be a fusion of airborne and scandoas SO2 gas flux measurements. Otherwise, the returned values will be the airborne SO2 gas flux measurements from cospec if available and contouring otherwise.


##### Parameters


`fuse: bool = ``False`  
If True, the returned values will be a fusion of airborne and scandoas SO2 gas flux measurements using a Kalman filter.

`smooth: bool = ``True`  
If True, the returned values will be smoothed using a Kalman smoother.


##### Returns


`pd.DataFrame`  
Dataframe with SO2 gas flux values.
