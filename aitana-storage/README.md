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

`aitana.storage.api.TonikAPI` exposes `/feature` and `/inventory` endpoints over
HTTP.

```bash
tonik_api --rootdir /data --port 8003
```

### Labelling in Grafana

To label what a detector flagged, draw on any dashboard: Ctrl/Cmd + click on a time
series panel marks a point, Ctrl/Cmd + click and drag marks a span. Tag it:

- `label:<class>` makes the annotation a label, e.g. `label:noise`, `label:instrument`,
  `label:tremor`. Annotations without one are ignored.
- `volcano:<name>`, e.g. `volcano:ruapehu`. Labels belong to a volcano, not to the
  dashboard they were drawn on.
- `series:<site>.<loc>.<cha>.<feature>`, only for a label about particular series rather
  than the whole volcano, e.g. an instrument fault: `series:FWVZ.10.HHZ.rsam`. Add one
  per series. The address matches the storage tree (`ruapehu/FWVZ/10/HHZ/rsam`). Scope by
  tag rather than by panel, because panel ids change when a dashboard is rebuilt.

To show a volcano's labels on a dashboard, whichever dashboard they were drawn on, set its
built-in *Annotations & Alerts* query to *Filter by: Tags* with `volcano:ruapehu` (or
`volcano:$volcano` on a templated dashboard). Change the built-in query rather than adding
a second one, or labels drawn on that dashboard show twice. Check once that Ctrl/Cmd +
click still adds annotations with that setting.

Then sync all labels, from every dashboard and volcano, with one label file. Run it on a
schedule and after every Grafana deploy:

```bash
grafana_annotations http://<user>:<pwd>@<grafana-host>:<port> --file /data/labels.json
```

The file is a JSON list ordered by time. Each label has `time` and `timeEnd` (ms since
epoch; `timeEnd` is null for a point), `title`, `description`, `tags`, a stable `id`, and
where it was made (`user`, `dashboardUID`, `panelId`).

Labels are human judgements only. What a detector decides is stored as series next to its
index, e.g. `rsam_mp_alarm` and `rsam_mp_threshold`, not as labels.

- Each label gets a stable id, kept as an `Id: <id>` line at the end of the annotation
  text; the sync appends it to new annotations. Leave that line in place when editing.
- The sync is a union. Edits to text, tags and times in Grafana win, and new labels are
  added to the file.
- Stored labels missing from Grafana are posted back as organization annotations (tied to
  no dashboard), so a freshly deployed Grafana gets every label back on the first sync.
  They show on every dashboard whose annotation query filters on the volcano tag.
- Deleting an annotation in Grafana therefore doesn't stick. To withdraw a label, retag it
  (e.g. `label:retracted`) and filter it out downstream.
- Keep annotation retention off (`max_age` and `max_annotations_to_keep` under
  `[annotations.dashboard]` and `[annotations.api]`), or the sync re-posts every aged-out
  label.
- Keep a `label:` tag on every label. An annotation that loses it no longer counts, and
  the sync posts the stored copy back.
- Labels with no `volcano:` tag are stored, and listed after the sync, because no volcano
  dashboard shows them.

The command prints how often each tag occurs, which shows typos such as `label:eruptoin`.

## License

See the repository root for license information.
