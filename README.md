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
  validation, gap-filling, and instrument response removal (`seismic_waveforms.py`); moving-window
  earthquake rate and gradient estimation (`util.py`).
- **Evaluation** — ROC curves, threshold evaluation (TP/FP/TN/FN), and forecasted-rate
  computation with pre-eruption windows (`scoring.py`).
- **State-space models** — Kalman-filter models for multi-sensor SO2 fusion and
  trend estimation (`assimilate.py`).
- **Benchmarking CLI** — `volcanobench` drives Snakemake workflows for end-to-end
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
- tonik

## Installation

```
pip install -U aitana
```

## Documentation

Learn more in the official [documentation](https://tsc-tools.github.io/aitana/).  
Try out a [Jupyter notebook](https://github.com/tsc-tools/aitana/blob/main/docs/aitana_example.ipynb).

## `volcanobench` CLI

The `volcanobench` command drives Snakemake benchmarking workflows bundled with Aitana
or registered by third-party packages via the `volcanobench.workflows` entry-point group.

### Commands

| Command | Description |
|---|---|
| `volcanobench list` | List all registered workflows (name, volcano, description). |
| `volcanobench run <volcano> <outdir>` | Run every registered workflow for a volcano. |
| `volcanobench clean <volcano> <outdir>` | Delete all outputs produced by the workflow. |

`volcanobench run` accepts `--cores N` (default `1`) to control Snakemake parallelism.

### Examples

```bash
# See what workflows are available
volcanobench list

# Run the Whakaari Bayesian-network benchmark, writing results to ./results
volcanobench run whakaari ./results

# Run with 4 parallel cores
volcanobench run whakaari ./results --cores 4

# Remove all outputs
volcanobench clean whakaari ./results
```

### Bundled workflows

| Name | Volcano | Description |
|---|---|---|
| `egu2026` | whakaari | Bayesian network vs decision tree benchmark for Whakaari/White Island (EGU 2026) |

### Registering your own workflow

Any Python package can register a workflow by exposing a `WorkflowDescriptor` instance
under the `volcanobench.workflows` entry-point group:

```toml
# pyproject.toml
[project.entry-points."volcanobench.workflows"]
my_workflow = "my_package.workflows:my_workflow"
```

```python
# my_package/workflows.py
from pathlib import Path
from aitana import WorkflowDescriptor

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

The workflow directory must contain a `Snakefile`. Aitana copies it into `outdir` before
execution, so the original is never modified.

## Get in touch

Report bugs, suggest features, view the source code, and ask questions
[on GitHub](https://github.com/tsc-tools/aitana/issues).
