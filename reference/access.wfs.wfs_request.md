## access.wfs.wfs_request()


Load volcano-tectonic earthquake catalogues from GeoNet's WFS service (http://wfs.geonet.org.nz).


Usage

``` python
access.wfs.wfs_request(
    start_date,
    end_date,
    polygon=None,
    radius=None,
    center_point=None,
    maxdepth=30
)
```


## Parameters

    polygon : str
        A string with the polygon coordinates in the format
        "lon1+lat1,lon2+lat2,...,lonN+latN,lon1+lat1"
    radius : int
        The radius around the volcano summit in meters.
    center_point : str
        The center point of the radius in the format "lon+lat".
    maxdepth : int
        The maximum depth of the earthquakes in kilometers.


## Returns

    pandas.DataFrame
        A dataframe with the earthquake catalogue.
