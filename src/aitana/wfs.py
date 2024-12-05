from datetime import datetime
import pandas as pd


def wfs_request(
    startdate: datetime,
    enddate: datetime,
    polygon: str = None,
    radius: int = None,
    center_point: str = None,
    maxdepth: int = 30
) -> pd.DataFrame:
    """
    Load volcano-tectonic earthquake catalogues from GeoNet's WFS service (http://wfs.geonet.org.nz).

    Parameters:
    -----------
        polygon : str
            A string with the polygon coordinates in the format
            "lon1+lat1,lon2+lat2,...,lonN+latN,lon1+lat1"
        radius : int
            The radius around the volcano summit in meters.
        center_point : str
            The center point of the radius in the format "lon+lat".
        maxdepth : int
            The maximum depth of the earthquakes in kilometers.

    Returns:
    --------
        pandas.DataFrame
            A dataframe with the earthquake catalogue.
    """
    wfs_url = "http://wfs.geonet.org.nz/geonet/ows?service=WFS&version=1.0.0"
    if polygon is None and radius is None:
        raise ValueError("Either polygon or radius must be provided.")
    startdate = startdate.strftime("%Y-%m-%dT%H:%M:%S.0Z")
    url = wfs_url
    url += (
        "&request=GetFeature&typeName=geonet:quake_search_v1&outputFormat=csv"
    )
    url += f"&cql_filter=origintime>={startdate}"

    if radius is not None:
        url += (
            f"+AND+DWITHIN(origin_geom,Point+({center_point}),{radius},meters)+AND+depth<{maxdepth}"
        )
    if polygon is not None:
        url += f"+AND+WITHIN(origin_geom,POLYGON(({polygon})))+AND+depth<{maxdepth}"

    cat = pd.read_csv(url, parse_dates=["origintime"])
    cat.sort_values(["origintime"], ascending=True, inplace=True)
    cat = cat.reset_index()
    cat = cat[cat.origintime <= str(enddate)]
    if cat.size == 0:
        msg = "There are no earthquakes available for"
        msg += f"the selected date range ({str(startdate)}, {str(enddate)}) in "
        if radius is not None:
            msg += "a radius of {} m".format(radius)
        if polygon is not None:
            msg += "the selected region: {}".format(polygon)
        raise ValueError(msg)

    return cat
