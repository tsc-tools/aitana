## volcanoes.ruapehu.Seismicity


Usage

``` python
volcanoes.ruapehu.Seismicity(
    start_date,
    end_date,
)
```


## Methods

| Name | Description |
|----|----|
| [cone()](#cone) | Load an earthquake catalogue for an area of 7 km around the summit. |
| [daily_rsam()](#daily_rsam) | Load RSAM values from DRZ, MAVZ and FWVZ combined by scaling |
| [regional()](#regional) | Load an earthquake catalogue for an area containing Waiouru and National Park. |
| [rm_duplicates()](#rm_duplicates) | Remove events from catalogue cat1 that are also in cat2 and return a new catalogue. |

------------------------------------------------------------------------


#### cone()


Load an earthquake catalogue for an area of 7 km around the summit.


Usage

``` python
cone()
```


------------------------------------------------------------------------


#### daily_rsam()


Load RSAM values from DRZ, MAVZ and FWVZ combined by scaling


Usage

``` python
daily_rsam(update=False)
```


MAVZ and FWVZ RSAM values with DRZ.


##### Returns

    :param df: Dataframe with RSAM values.
    :type df: pandas.DataFrame

------------------------------------------------------------------------


#### regional()


Load an earthquake catalogue for an area containing Waiouru and National Park.


Usage

``` python
regional()
```


------------------------------------------------------------------------


#### rm_duplicates()


Remove events from catalogue cat1 that are also in cat2 and return a new catalogue.


Usage

``` python
rm_duplicates(cat1, cat2)
```


##### Parameters

    cat1 : pandas.DataFrame
        The first catalogue.
    cat2 : pandas.DataFrame
        The second catalogue.


##### Returns

    pandas.DataFrame
        A new catalogue with the events from cat1 that are not in cat2.
