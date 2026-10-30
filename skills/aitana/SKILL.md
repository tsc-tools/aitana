---
name: aitana
description: >
  Write Python code with Aitana, the `aitana.*` namespace packages for volcano monitoring:
  GeoNet Tilde and WFS data, seismic waveforms from GeoNet's S3 archive and FDSN services,
  RSAM, SSAM, DSAR and spectral features, NetCDF/Zarr feature storage, anomaly indices
  (matrix profile), detector evaluation with synthetic faults, and Snakemake workflows run by
  `aitana-pipelines`. Use this skill whenever code imports `aitana.` or the user mentions an
  `aitana-*` package, and also when they want to fetch New Zealand volcano monitoring data from
  GeoNet, compute seismic features such as RSAM from waveforms, store or query feature time
  series, or score volcanic unrest, even if they don't name Aitana. Aitana also provides utilities to
  generate graphs from raw data, feature time series, and anomaly scores.
---

# Aitana

Aitana is a set of small Python distributions that share one import namespace, `aitana.`.
Each installs on its own with its own dependencies, and they are built to be combined: fetch
data, compute features, store them, score them for anomalies, and run the chain as a
reproducible workflow. This skill is for code that *uses* the packages. For changing the
Aitana repository itself, follow its developer notes instead:
https://tsc-tools.github.io/aitana/developer/architecture.html

## Install the right distributions

There is no `aitana` distribution that installs everything. `pip install aitana` installs an
unrelated pre-split release (0.2.0, data files only), so don't suggest it. Pick the
distributions for the job; their own `aitana-*` dependencies come along automatically.

| You need | Install | Import from |
|---|---|---|
| Tilde and WFS data, seismic waveforms | `aitana-access` | `aitana.access.tilde`, `.wfs`, `.seismic` |
| RSAM, SSAM, DSAR, spectral features | `aitana-features` | `aitana.features` |
| Store and query feature time series, HTTP API | `aitana-storage` | `aitana.storage` |
| Ready-made Ruapehu/Whakaari datasets, eruption catalogues | `aitana-volcanoes` | `aitana.volcanoes.ruapehu`, `.whakaari` |
| Synthetic test waveforms | `aitana-testing` | `aitana.testing` |
| Anomaly detectors, evaluation harness | `aitana-anomaly` | `aitana.anomaly.<module>` |
| `aitana-pipelines` CLI, scoring, workflows | `aitana-pipelines` | `aitana.pipelines.<module>` |


- Extras: `aitana-anomaly[pytorch]` for `AutoEncoder`, and `aitana-storage[pytorch]` for
  `to_pytorch()` on a store or substore.
- `import aitana` gives an empty namespace; always import a sub-package. Only
  `aitana.features` (`RSAM`, `SSAM`, `DSAR`, `SpectralFeatures`), `aitana.storage` (`Storage`,
  `generate_test_data`) and `aitana.testing` (`synthetic_trace`) re-export names. Elsewhere,
  import from the module, e.g. `from aitana.access.seismic import SeismicData`.

## How the pieces fit

```
aitana.access     fetch    -> pandas DataFrames (Tilde, WFS), ObsPy Traces (waveforms)
aitana.features   compute  -> xarray Datasets indexed by "datetime"
aitana.storage    store    <- xarray Datasets with a "datetime" dimension
aitana.anomaly    score    -> pandas Series in [0, 1]
aitana.pipelines  run all of the above as Snakemake workflows
```

Keep fetching and computing apart, as the packages do. Feature classes take an in-memory ObsPy
`Trace` and never download anything, so fetch with `aitana.access` and pass the trace in. That
also lets feature code be tested offline with `aitana.testing.synthetic_trace`.

## Waveforms to features

```python
from obspy import UTCDateTime
from aitana.access.seismic import FDSNWaveforms, S3Waveforms, SeismicData
from aitana.features import RSAM

# GeoNet's S3 archive lags about a day behind; the near-real-time FDSN service covers recent days.
sd = SeismicData([S3Waveforms(), FDSNWaveforms("https://service-nrt.geonet.org.nz")])
tr = next(sd.get_waveforms("NZ", "FWVZ", "10", "HHZ",
                           UTCDateTime(2022, 3, 13), UTCDateTime(2022, 3, 14)))
if len(tr) > 0:
    rsam = RSAM(interval=600, max_gap_fraction=0.1).compute(tr)  # xarray Dataset
```

- Clients are tried in order and the first with data wins. Returned waveforms have the
  instrument sensitivity removed, gaps filled with NaN, and are trimmed to the request.
- `get_waveforms` is a generator. Without `chunk_size` it yields one `Trace` for the whole span,
  hence `next(...)`; with `SeismicData(..., chunk_size=<seconds>)` it yields one per chunk. A
  channel wildcard such as `HH?` yields a `Stream` with Z/N/E (or Z/1/2) instead.
- Missing data gives an empty `Trace`, not an exception, so check `len(tr)`.
- `get_waveforms(..., cache=True)` keeps raw day files in `~/aitana_cache/raw` (SDS layout)
  for UTC days that ended more than two days ago.
- `max_gap_fraction` sets windows with too much missing data to NaN instead of computing
  across interpolated gaps.

Every feature class has a `compute(trace)` method that returns an xarray `Dataset`:

| Class | Defaults | Data variables |
|---|---|---|
| `RSAM` | 600 s windows, 2–5 Hz | `rsam`, `rsam_energy_prob` |
| `DSAR` | 600 s windows, 4.5–8 Hz over 8–16 Hz | `dsar` |
| `SSAM` | **10 s** windows | `ssam` (frequency × datetime), `filterbank`, `sonogram` |
| `SpectralFeatures` | 600 s windows, 0.5 Hz high-pass | `central_freq`, `bandwidth`, `predom_freq` |

Pass `interval` explicitly for `SSAM`: its 10 s default differs from the others. The RSAM
energy variable really is `rsam_energy_prob`, although `RSAM.features` lists
`rsam_energy_prop`. Use `.to_series()` on a variable to get a pandas Series.

## Storing and querying features

```python
from datetime import datetime
from aitana.storage import Storage

root = Storage("ruapehu", "/data/aitana")          # NetCDF by default; backend="zarr" for Zarr
store = root.get_substore("FWVZ", "10", "HHZ")     # e.g. site / location / channel
store.save(rsam)                                   # Dataset with a "datetime" dimension

root.starttime = datetime(2022, 3, 1)
root.endtime = datetime(2022, 3, 31)
series = store("rsam")                             # DataArray sliced to the root's time range
```

- Each data variable becomes its own feature archive, and saving again merges new data in.
  Archives are padded with NaN back to `archive_starttime` (default 2000-01-01, set on the
  `Storage`), so a query returns NaN wherever nothing was saved; drop them with
  `series.dropna("datetime")` when that matters.
- Set `starttime` and `endtime` on the `Storage` before querying. They apply to its
  substores, including ones created earlier; without them a query raises `TypeError`.
- Query by calling the substore with the feature name. `store.load("rsam")` currently returns
  `None`.
- To save a pandas Series, convert it first:
  `s.rename_axis("datetime").to_xarray().to_dataset(name="my_feature")`.
- `aitana-api --rootdir /data/aitana --port 8003` serves stored features over HTTP (`/feature`,
  `/inventory`), e.g. for Grafana's Infinity data source.

## Tilde and WFS

```python
from datetime import datetime, timezone
from aitana.access.tilde import tilde_request
from aitana.access.wfs import wfs_request

start = datetime(2019, 1, 1, tzinfo=timezone.utc)
end = datetime(2019, 12, 31, tzinfo=timezone.utc)
so2 = tilde_request(start, end, domain="manualcollect", name="plume-SO2-gasflux",
                    station="WI000", method="contouring", sensor="MC01")  # columns obs, err; index dt

quakes = wfs_request(start, end, radius=20000, center_point="177.186833+-37.523118", maxdepth=30)
quakes = quakes.loc[:end]  # wfs_request does not apply end_date
```

- Pass timezone-aware UTC datetimes. Naive ones are read as local time, which in New Zealand
  shifts the window by 12–13 hours.
- Look up Tilde `domain`, `name`, `station`, `method` and `sensor` codes in the data discovery
  tool: https://tilde.geonet.org.nz/ui/data-discovery/. A valid stream can come back empty
  for the requested range; HTTP errors raise `ValueError`.
- `wfs_request` returns every event from `start_date` to now, so slice the result. It raises
  `ValueError` when nothing matches. `center_point` is `"lon+lat"`, `radius` is in metres,
  `maxdepth` in km; `polygon="lon1+lat1,lon2+lat2,...,lon1+lat1"` is the alternative to a radius.

## Ready-made volcano datasets

```python
from datetime import datetime, timezone
from aitana.volcanoes import whakaari

df = whakaari.load_all(end_date=datetime.now(timezone.utc))  # daily Eqr, RSAM, CO2, H2S, SO2
eruptions = whakaari.eruptions()                             # eruption catalogue, indexed by date
```

- `ruapehu` has the same `load_all` and `eruptions`, plus `CraterLake(start, end)` with
  `temperature()`, `water_level()` and `water_analyte()`. Both modules have `Gas` and
  `Seismicity` classes for the individual streams.
- These download from Tilde and WFS (and Whakaari RSAM from Zenodo), so the first call is slow.
  Results are cached as CSV files in `~/.aitana_cache`; delete them to force a refetch.
- Pass `end_date` explicitly. The default `datetime.now(...)` is evaluated once at import, so
  in a long-running process it goes stale.

## Anomaly indices

```python
import numpy as np
import pandas as pd
from aitana.anomaly.matrix_profile import MatrixProfile

x = np.log10(series.to_series())            # regular 10-min samples: history, then the period to score
detector = MatrixProfile(window=36)         # subsequence length in samples: 6 h of 10-min data
index = detector.compute_anomaly_index(x, score_start=pd.Timestamp("2022-03-02"))
```

- Everything before `score_start` is the reference history. The index is the percentile rank,
  in [0, 1], of each new subsequence's distance to its nearest *earlier* subsequence. It is NaN
  before `score_start` and wherever a subsequence contains NaN; `detector.profile_` holds the raw
  distances. A series without finite history before `score_start` raises `ValueError`.
- It is z-normalized, so it reacts to changes in shape, not level. Pair it with an amplitude
  threshold when level matters.
- `MatrixProfile` is the only detector that produces an anomaly index today. `SNet` computes
  scattering-network coefficients (a feature, not a score), `PCA` is not usable yet, and
  `AutoEncoder` needs the `pytorch` extra.
- For a detector of your own, subclass `aitana.anomaly.anomaly_base.AnomalyDetectionBaseClass`,
  implement `compute_anomaly_index`, and map raw scores to [0, 1] with
  `normalize_score(raw_scores, reference_scores)` so the index is comparable with the others.

## Evaluating a detector

Real labelled anomalies are rare, so evaluate against synthetic faults with known positions:

```python
from aitana.anomaly.evaluation import inject_faults, score_detector

corrupted, mask = inject_faults(df, columns=["rsam"], rate=0.06)  # mask: True where a fault was injected
index = my_detector(corrupted)                                    # per-timestep scores in [0, 1]
ok = index.notna()                                                # score_detector rejects NaN
result = score_detector(index[ok], mask[ok], threshold=0.9)       # fpr, tpr, auc, tp, fp, tn, fn
```

Anomaly indices are NaN before `score_start` and around gaps, so drop those timesteps (from
both the index and the mask) before scoring.

`inject_change_points` (persistent regime shifts), `simulate_regime_series` (data with known
states) and `quiet_series` cover other cases. Worked examples:
https://tsc-tools.github.io/aitana/user-guide/evaluating-detectors.html

## Workflows

```bash
aitana-pipelines list
aitana-pipelines run ruapehu ~/rsam_mp --workflow ruapehu_mp --cores 4 --config end=2022-03-20T00:00
aitana-pipelines clean ruapehu ~/rsam_mp --workflow ruapehu_mp
```

- The bundled workflows are `egu2026` (Whakaari), `ruapehu_mp` and `ruapehu_snet`. Pick one with
  `--workflow` when a volcano has several, and give each its own output directory.
- Extra arguments go to Snakemake (`--config key=value`, `-n` for a dry run). `--cron "<expr>"`
  keeps running on a UTC crontab schedule.
- Workflow rules run in the active environment, so install every package their scripts import
  (the install command above covers the bundled ones).
- Register your own workflow under the `aitana.pipelines.workflows` entry-point group with an
  `aitana.pipelines.cli.WorkflowDescriptor(name, volcano, description, workflowdir, outputs)`;
  `workflowdir` must contain a `Snakefile`. Details:
  https://tsc-tools.github.io/aitana/user-guide/running-workflows.html

## Testing without the network

`aitana.testing.synthetic_trace(nsec=3600, sampling_rate=100.0, ...)` returns an ObsPy `Trace`
of sinusoids at known frequencies (0.1, 3 and 10 Hz by default) plus noise and a chirp; set
`starttime`, `station` etc. to match real data and `gaps=True` to exercise gap handling.
`aitana.storage.generate_test_data(dim=1)` returns a Dataset ready for `Storage.save`.

## Looking things up

- API reference, one page per class and function:
  https://tsc-tools.github.io/aitana/reference/
- Some classes have no class docstring (e.g. `SeismicData`, `SDSWaveforms`). Their parameters
  are often documented on `__init__` instead; use `help(Class.__init__)`, or read the source.
