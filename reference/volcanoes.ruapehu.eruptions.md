## volcanoes.ruapehu.eruptions()


This function loads the eruption catalogue for Mt. Ruapehu and declusters


Usage

``` python
volcanoes.ruapehu.eruptions(
    min_size=0,
    dec_interval=None,
)
```


it if required. The catalogue is based on Brad Scott's GNS Science Report. Declustering is turned on if dec_interval is greater than zero. This will also exclude magmatic eruption episodes between 9 January 1944 and 8 January 1946 as well as 6 January 1995 to 12 January 1997 as these aren't independent events.


## Parameters

    min_size : int
        The minimum eruption size to include in the catalogue.
    dec_interval : str
        The declustering interval in pandas complient time delta format.


## Returns

    pandas.DataFrame
        A dataframe with the (declustered) eruption catalogue.
