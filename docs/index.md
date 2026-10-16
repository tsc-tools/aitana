# Aitana: volcano monitoring benchmarking framework

Aitana (named after the great football midfielder
[Aitana Bonmati](https://en.wikipedia.org/wiki/Aitana_Bonmat%C3%AD)) is a Python library
intended as a prototype implementation of a validation and benchmarking framework for
AI/ML in volcano monitoring.

Operational uptake of machine learning in volcano observatories remains limited because of
three structural problems: no community-accepted benchmark datasets, poor reproducibility,
and insufficient uncertainty quantification. Aitana is a prototype addressing these gaps.

## What is implemented

- **Data access** — time-series data from three New Zealand volcanoes
  ([Ruapehu](https://volcano.si.edu/volcano.cfm?vn=241100),
  [Whakaari/White Island](https://volcano.si.edu/volcano.cfm?vn=241040),
  [Taupo](https://volcano.si.edu/volcano.cfm?vn=241070)) via the GeoNet TILDE API and WFS,
  with transparent disk caching and incremental date-range updates.
- **Pre-processing** — RSAM access for Ruapehu and Whakaari; seismic waveform retrieval,
  validation, gap-filling, and instrument response removal (`seismic_waveforms`); moving-window
  earthquake rate and gradient estimation (`util`).
- **Evaluation** — ROC curves, threshold evaluation (TP/FP/TN/FN), and forecasted-rate
  computation with pre-eruption windows (`scoring`).
- **State-space models** — Kalman-filter models for multi-sensor SO₂ fusion and
  trend estimation (`assimilate`).
- **Workflow CLI** — `aitana-pipelines` drives Snakemake workflows for end-to-end
  benchmarking (download → feature extraction → model training → forecast scoring).
  One workflow is bundled: a Bayesian network vs decision tree benchmark for Whakaari/White Island.

## What is planned / in progress

- Shared implementations of DSAR, spectrograms, and swarm detection.
- Temporal cross-validation schemes.
- Proper scoring rules (log-likelihood, CRPS, reliability diagrams).
- Containerised or environment-locked pipelines (Docker / Apptainer / pixi).
- Semantic versioning of datasets and evaluation protocols.
- Integration with community tools (SeisBench, WOVOdat formats).

## Dependencies

- pandas, requests, matplotlib, statsmodels
- obspy (seismic waveform processing)
- snakemake (workflow execution)
- aitana-storage

## Installation

```
pip install -U aitana
```

## `aitana-pipelines` CLI

The `aitana-pipelines` command drives Snakemake workflows bundled with Aitana or
registered by third-party packages via the `aitana.pipelines.workflows` entry-point group.

| Command | Description |
|---|---|
| `aitana-pipelines list` | List all registered workflows (name, volcano, description). |
| `aitana-pipelines run <volcano> <outdir>` | Run every registered workflow for a volcano. |
| `aitana-pipelines clean <volcano> <outdir>` | Delete all outputs produced by the workflow. |

`aitana-pipelines run` accepts `--cores N` (default `1`) to control Snakemake parallelism.

## Get in touch

Report bugs, suggest features, view the source code, and ask questions [on GitHub](https://github.com/tsc-tools/aitana/issues).
