## features.util.apply_freq_filter()


Apply frequency filter to seismic data.


Usage

``` python
features.util.apply_freq_filter(
    data, filtertype, filterfreq, corners=4, zerophase=False, **options
)
```


Filter parameters are taken from instance attributes `filtertype` and `filterfreq`.

## Parameters

- **data** : `obspy.core.trace.Trace```  or :class:` ```obspy.core.stream.Stream` -- The seismic data to be filtered.
- **corners** : `int` -- Filter corners / order, defaults to 4.
- **zerophase** : `bool` -- If True, apply filter once forwards and once backwards. Defaults to False.
- **options** -- Passed to underlying obspy filter methods.
