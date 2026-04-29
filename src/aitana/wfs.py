from urllib.error import HTTPError
from datetime import datetime
import pandas as pd
from aitana.logging_config import get_logger

logger = get_logger(__name__)


def wfs_request(
    start_date: datetime,
    end_date: datetime,
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
    start_date = start_date.strftime("%Y-%m-%dT%H:%M:%S.0Z")
    url = wfs_url
    url += (
        "&request=GetFeature&typeName=geonet:quake_search_v1&outputFormat=csv"
    )
    url += f"&cql_filter=origintime>={start_date}"

    if radius is not None:
        url += (
            f"+AND+DWITHIN(origin_geom,Point+({center_point}),{radius},meters)+AND+depth<{maxdepth}"
        )
    if polygon is not None:
        url += f"+AND+WITHIN(origin_geom,POLYGON(({polygon})))+AND+depth<{maxdepth}"
    try:
        cat = pd.read_csv(url, parse_dates=[
                          "origintime"], index_col="origintime")
    except HTTPError as e:
        logger.error(f"HTTP Error: {e.code} - {e.reason}")
        logger.error("URL: " + url)
        raise e
    cat.sort_index(inplace=True)
    if cat.size == 0:
        msg = "There are no earthquakes available for"
        msg += f"the selected date range ({str(start_date)}, {str(end_date)}) in "
        if radius is not None:
            msg += "a radius of {} m".format(radius)
        if polygon is not None:
            msg += "the selected region: {}".format(polygon)
        raise ValueError(msg)

    return cat
