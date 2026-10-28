# Running workflows

The `aitana-pipelines` command drives Snakemake workflows bundled with Aitana or registered by third-party packages via the `aitana.pipelines.workflows` entry-point group.


# Commands

| Command | Description |
|----|----|
| `aitana-pipelines list` | List all registered workflows (name, volcano, description). |
| `aitana-pipelines run <volcano> <outdir>` | Run the workflow registered for a volcano. |
| `aitana-pipelines clean <volcano> <outdir>` | Delete all outputs produced by the workflow. |

Where a volcano has several workflows, choose one with `--workflow <name>`; each needs its own `<outdir>`, since a run copies the workflow's `Snakefile` and `config.yaml` into it.

`aitana-pipelines run` accepts `--cores N` (default `1`) to control Snakemake parallelism. With `--cron "<crontab expression>"` it keeps running and repeats on that schedule (UTC), skipping a run that is due while the previous one is still going. Any further arguments are passed to Snakemake, e.g. `--config end=2022-03-20T00:00` or `-n` for a dry run.


# Examples

``` bash
# See what workflows are available
aitana-pipelines list

# Run the Whakaari Bayesian-network benchmark, writing results to ./results
aitana-pipelines run whakaari ./results

# Run with 4 parallel cores
aitana-pipelines run whakaari ./results --cores 4

# Remove all outputs
aitana-pipelines clean whakaari ./results
```


# Bundled workflows

| Name | Volcano | Description |
|----|----|----|
| `egu2026` | whakaari | Bayesian network vs decision tree benchmark for Whakaari/White Island (EGU 2026) |
| `ruapehu_mp` | ruapehu | Matrix-profile anomaly index and alarm series of Ruapehu RSAM, stored for Grafana |
| `ruapehu_snet` | ruapehu | Scattering-network coefficients of three-component Ruapehu seismic data, stored for detectors |


# The Ruapehu RSAM alert, end to end

`ruapehu_mp` fetches `NZ.FWVZ.10.HHZ` waveforms (GeoNet's S3 archive, falling back to the near-real-time FDSN service for the last day or two), computes 10-min RSAM per UTC day, scores the last 24 h against the preceding 28 days with [MatrixProfile](../reference/anomaly.matrix_profile.MatrixProfile.md#aitana.anomaly.matrix_profile.MatrixProfile), and stores RSAM ([rsam](../reference/volcanoes.whakaari.Seismicity.md#aitana.volcanoes.whakaari.Seismicity.rsam)), the anomaly index (`rsam_mp_index`), the raw left matrix profile it is the percentile of (`rsam_mp_profile`: each 6-h window's z-normalized distance to its nearest earlier window, 0 to 12), its alarm series (`rsam_mp_alarm`: 1 where the index reaches `threshold`, 0 below, NaN where there is no index) and the threshold in force when each sample was scored (`rsam_mp_threshold`, so retunes stay traceable and Grafana can draw it on the index panel) in `aitana-storage` under `<outdir>/storage`. Each run also writes `results/<end>/index.csv` with RSAM, index, matrix profile and alarm. Completed days are reused across runs. Its rules run in the active environment, so run it from the uv workspace (`uv sync`). With `--config start=2022-01-01`, the first run also builds the archive: it scores every complete UTC day since `start` exactly as a run ending at the next midnight would, and writes `results/archive_<start>.csv` instead. Later runs find that file and only score the last 24 h; delete it to rebuild the archive, e.g. after a retune or an outage of more than a day.

``` bash
# Replay a past day (first run downloads ~29 days of waveforms, ~250 MB)
aitana-pipelines run ruapehu ~/rsam_mp --workflow ruapehu_mp --cores 4 --config end=2022-03-20T00:00

# Archive since 2022-01-01 (streams ~14 GB of waveforms; keeps ~10 MB of RSAM)
aitana-pipelines run ruapehu ~/rsam_mp --workflow ruapehu_mp --cores 8 --config start=2022-01-01

# Keep scoring every 10 min and serve the stored series for Grafana
aitana-pipelines run ruapehu ~/rsam_mp --workflow ruapehu_mp --cron "2-59/10 * * * *" &
aitana-api --rootdir ~/rsam_mp/storage --port 8003
```

Grafana reads the series from `aitana-api` with the Infinity data source (CSV, **Backend** parser, which alerting requires), e.g. `/feature?path=ruapehu,FWVZ,10,HHZ&name=rsam_mp_alarm&starttime=${__from:date:iso}&endtime=${__to:date:iso}`. An alert rule on the maximum of `rsam_mp_alarm` posts to Slack, and a No Data rule on `rsam_mp_index` catches a stalled pipeline (or a station data gap).

On the default branch, [.gitlab-ci.yml](https://github.com/tsc-tools/aitana/blob/main/.gitlab-ci.yml) deploys these two commands as the `aitana-ruapehu-mp` and `aitana-ruapehu-api` containers, plus `aitana-label-sync`, which syncs the labels drawn in Grafana with `labels.json` every 10 minutes (see [aitana-storage](https://github.com/tsc-tools/aitana/blob/main/aitana-storage/README.md)). All three share the data directory `DATA_VOLUME`.

The z-normalized matrix profile responds to changes in RSAM *shape*, not level; an amplitude threshold is a separate alert. Over Jan-Jun 2022 it alarmed on 1-3 days per month in quiet months and on about 20 % of days during the March-May 2022 unrest. It did not flag the abrupt 2022-03-13 onset itself (a step looks like any other step once z-normalized); the first alarm after it was 2022-03-19 17:30 UTC, on an unusually steady stretch of tremor.


# Scattering coefficients of Ruapehu

`ruapehu_snet` fetches `NZ.FWVZ.10.HH?` (Z, N, E) the same way, resamples it to 25 Hz with an anti-alias filter and computes first- and second-order scattering coefficients with [SNet](../reference/anomaly.snet.SNet.md#aitana.anomaly.snet.SNet) (20-min windows every 10 min, each labelled by its first sample). It caches one file per UTC day under `snet/` and stores `snet_order1` (datetime, channel, frequency) and `snet_order2` (datetime, channel, frequency, order2 frequency) under `ruapehu/FWVZ/10/HH` in `<outdir>/storage`; channel 0 is Z. Each run plots the vertical first-order coefficients to `results/<end>/order1.png`. A run covers the last 24 h, or every day since `start`. A day takes about 2.5 CPU-minutes and ~27 MB of waveforms. No detector consumes the coefficients yet.

``` bash
# Replay the 2022-03-13 onset
aitana-pipelines run ruapehu ~/snet --workflow ruapehu_snet --cores 4 --config end=2022-03-14T05:30

# Backfill since 2022-01-01
aitana-pipelines run ruapehu ~/snet --workflow ruapehu_snet --cores 8 --config start=2022-01-01
```


# Running the EGU 2026 workflow with Docker

`scripts/run_egu2026.sh` is a convenience wrapper that builds the project's Docker image, installs Aitana from source inside the container, and runs the bundled `egu2026` workflow -- no local Python environment needed.

**Prerequisites:** Docker must be installed and running.

``` bash
# Run with defaults (output → ./egu_2026_output, all available cores)
./scripts/run_egu2026.sh

# Custom output directory and core count
./scripts/run_egu2026.sh -o /data/results -c 8

# Show help
./scripts/run_egu2026.sh -h
```

| Option      | Default             | Description                    |
|-------------|---------------------|--------------------------------|
| `-o OUTDIR` | `./egu_2026_output` | Directory for workflow outputs |
| `-c CORES`  | all logical CPUs    | Snakemake parallelism          |

The Docker image (`aitana-workflow`) is built from `docker/Dockerfile` on first run and cached by Docker. Conda environments for individual workflow rules are cached in a named Docker volume (`aitana-conda-envs`) and reused across runs, so subsequent invocations are significantly faster.

The poster-generation step uses Apptainer inside Docker (to run an Inkscape container), which requires elevated capabilities (`SYS_ADMIN`, `/dev/fuse`). These are passed automatically by the script.


# Registering your own workflow

Any Python package can register a workflow by exposing a [WorkflowDescriptor](../reference/pipelines.cli.WorkflowDescriptor.md#aitana.pipelines.cli.WorkflowDescriptor) instance under the `aitana.pipelines.workflows` entry-point group:

``` toml
# pyproject.toml
[project.entry-points."aitana.pipelines.workflows"]
my_workflow = "my_package.workflows:my_workflow"
```

``` python
# my_package/workflows.py
from pathlib import Path
from aitana.pipelines.cli import WorkflowDescriptor

my_workflow = WorkflowDescriptor(
    name="my_workflow",
    volcano="ruapehu",
    description="My custom eruption forecast model",
    workflowdir=Path(__file__).parent / "snakemake",
    outputs={
        "forecast": "results/forecasts.nc",
    },
)
```

The workflow directory must contain a `Snakefile`. Aitana copies it into `outdir` before execution, so the original is never modified.
