## access.util.eqRate()


Compute earthquake rate.


Usage

``` python
access.util.eqRate(
    cat, fixed_time=None, fixed_nevents=None, enddate=datetime.utcnow()
)
```


## Parameters

- **cat** : `pandas.DataFrame` -- A catalogue of earthquakes as returned by :method:`pyvolprob.load_ruapehu_earthquakes`
- **fixed_time** : `int` -- If not 'None', compute the earthquake rate based on a fixed-length time window given in days.
- **fixed_nevents** : `int` -- If not None, compute the earthquake rate based on a fixed number of events.
- **enddate** : `datetime.datetime` -- The latest date of the time-series. Mainly needed for testing.
