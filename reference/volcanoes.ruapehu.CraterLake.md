## volcanoes.ruapehu.CraterLake


Usage

``` python
volcanoes.ruapehu.CraterLake(
    start_date,
    end_date,
)
```


## Methods

| Name | Description |
|----|----|
| [temperature()](#temperature) | Read crater lake temperature from Tilde (https://tilde.geonet.org.nz). |
| [water_analyte()](#water_analyte) | Download water analyte data from Tilde (https://tilde.geonet.org.nz). |
| [water_level()](#water_level) | Read crater lake water level from Tilde (https://tilde.geonet.org.nz). |

------------------------------------------------------------------------


#### temperature()


Read crater lake temperature from Tilde (https://tilde.geonet.org.nz).


Usage

``` python
temperature(resample="D", interpolate=None, exclude1995=True, dropna=True)
```


##### Parameters

    :param resample: Average over resampling interval and
                    linearly interpolate in between. The
                    interval should be one of 'D', 'W',
                    or 'M'
    :type resample: str
    :param interpolate: Interpolate between resampled points.
                        Only takes effect if resample is not
                        None.
    :type interpolate: str


##### Returns

    :param df: A dataframe with the temperature time series.
    :type df: pandas.Dataframe

------------------------------------------------------------------------


#### water_analyte()


Download water analyte data from Tilde (https://tilde.geonet.org.nz).


Usage

``` python
water_analyte(
    analyte,
    resample=None,
    interpolate=None,
    exclude1995=True,
    drop_duplicates=True
)
```


##### Parameters

    analyte : str
        The analyte to download. This can be one of
            'Al, 'As', 'B', 'Br', 'Ca', 'Cl', 'Cs', 'F',
            'Fe', 'H2S', 'K', 'Li', 'Mg', 'NH3', 'NO3-N', 'Na',
            'PO4-P', 'Rb', 'SO4', 'SiO2', 'd18O', 'd2H', 'ph'
    resample : str
        Resample the data to a given interval. This can be anything
        allowed by :pandas.DataFrame.resample:.
    interpolate : str
        Interpolate between points. This can be anything allowed by
        :pandas.DataFrame.interpolate:.
    exclude1995 : bool
        Exclude data from 1995 when there wasn't a lake.


##### Returns

    pandas.DataFrame
        A dataframe with the water analyte time series.

------------------------------------------------------------------------


#### water_level()


Read crater lake water level from Tilde (https://tilde.geonet.org.nz).


Usage

``` python
water_level(resample="D", interpolate=None, dropna=True)
```
