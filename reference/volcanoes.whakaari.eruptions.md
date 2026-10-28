## volcanoes.whakaari.eruptions()


This function loads the eruption catalogue for White Island and


Usage

``` python
volcanoes.whakaari.eruptions(
    min_size=0,
    dec_interval=None,
)
```


declusters it.

## Parameters

- **eruption_scale** : `int` -- The minimum eruption scale (between 0 and 4) to use. This is currently ignored as the White Island catalogue does not contain eruption scale.
- **dec_interval** : `pandas.DateOffset`` or str` -- The declustering interval, which is the minimum distance in time between any two eruptions.
- **datadir** : `str` -- Path that contains the catalogue file in csv format.

## Returns

`pandas.DataFrame`  
Declustered eruption catalogue
