# aitana-access

Pure I/O access layer for volcano monitoring data: GeoNet TILDE observations,
GeoNet WFS earthquake catalogues, and seismic/acoustic waveforms (local SDS
archive, FDSN web services, or GeoNet's open S3 data archive).

## Installation

```bash
pip install aitana-access
```

## Usage

### TILDE observations

```python
from datetime import datetime
from aitana.access.tilde import tilde_request

df = tilde_request(
    start_date=datetime(2019, 1, 1),
    end_date=datetime(2019, 1, 3),
    domain="envirosensor",
    name="lake-temperature",
    station="RU001",
    sensor="01",
)  # -> pandas.DataFrame indexed by datetime, columns ["obs", "err"]
```

### WFS earthquake catalogue

```python
from datetime import datetime
from aitana.access.wfs import wfs_request

cat = wfs_request(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2020, 2, 1),
    center_point="176.17+-39.28",
    radius=20000,   # metres
    maxdepth=30,    # km
)  # -> pandas.DataFrame indexed by origintime
```

Provide either `radius` + `center_point`, or `polygon` (as
`"lon1+lat1,lon2+lat2,...,lonN+latN,lon1+lat1"`).

### Seismic waveforms

`SeismicData` fetches waveforms from one or more clients (tried in order,
first success wins) and yields `obspy.Trace`/`Stream` objects with
instrument sensitivity removed.

```python
from obspy import UTCDateTime
from aitana.access.seismic import SeismicData, FDSNWaveforms, S3Waveforms

clients = [
    S3Waveforms(staxml_dir="./staxml"),
    FDSNWaveforms(url="https://service.geonet.org.nz"),
]
ds = SeismicData(clients)

for trace in ds.get_waveforms(
    net="NZ", site="WIZ", loc="10", comp="HHZ",
    start=UTCDateTime("2020-01-01"),
    end=UTCDateTime("2020-01-02"),
):
    print(trace)
```

Other waveform clients: `SDSWaveforms` (local SDS archive), `MockSDSWaveforms`
(synthetic data for testing). All implement `get_waveforms(net, site, loc, comp, startdate, enddate)`.

### FastAPI service

`aitana.access.api.AitanaAPI` exposes `/waveform` and `/spectrogram` endpoints
over HTTP.

```bash
python -m aitana.access.api --rootdir /data --port 8003
```

## License

See the repository root for license information.
