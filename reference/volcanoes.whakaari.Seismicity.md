## volcanoes.whakaari.Seismicity


Usage

``` python
volcanoes.whakaari.Seismicity(
    start_date,
    end_date,
)
```


## Methods

| Name | Description |
|----|----|
| [rsam()](#rsam) | Load RSAM values NZ.WIZ.10.HHZ.The first time you call |

------------------------------------------------------------------------


#### rsam()


Load RSAM values NZ.WIZ.10.HHZ.The first time you call


Usage

``` python
rsam()
```


this it will download the data from Zenodo and store it locally.


##### Returns

    :param df: Dataframe with RSAM values.
    :type df: pandas.DataFrame
