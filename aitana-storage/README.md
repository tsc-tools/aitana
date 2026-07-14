# aitana-storage

Store time series feature data as NetCDF or Zarr archives, organised in a
directory hierarchy, and access them through a FastAPI service.

## Installation

```bash
pip install aitana-storage
```

`to_pytorch()` requires PyTorch, which is an optional dependency:

```bash
pip install aitana-storage[pytorch]
```

## Usage

### Saving and querying features

`Storage` organises features in a directory tree (e.g. site/sensor/channel)
and stores each feature as a NetCDF or Zarr archive on disk.

```python
from datetime import datetime
from aitana.storage import Storage, generate_test_data

g = Storage('Whakaari', '/tmp', backend='zarr')
c = g.get_substore('WIZ', '00', 'HHZ')
c.save(generate_test_data(dim=1))

g.starttime = datetime(2012, 1, 1)
g.endtime = datetime(2012, 1, 2, 23, 59, 59)
rsam = c('rsam')  # -> xarray.DataArray indexed by datetime
```

### PyTorch dataset

```python
from torch.utils.data import DataLoader

dataset = g.to_pytorch(['rsam', 'dsar'])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### FastAPI service

`aitana.storage.api.TonikAPI` exposes `/feature`, `/inventory` and `/labels`
endpoints over HTTP.

```bash
tonik_api --rootdir /data --port 8003
```

### Grafana annotations

Post labels stored alongside features as Grafana annotations, or retrieve
existing annotations from the Grafana API.

```bash
grafana_annotations http://<user>:<pwd>@<grafana-host>:<port> --labels labels.json
```

## License

See the repository root for license information.
