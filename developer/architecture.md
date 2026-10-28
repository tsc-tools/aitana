# Architecture

Package boundaries are drawn along the **axis of change** -- who has to change when X changes, and which way dependencies point -- not along a taxonomy of what the data is. Two rules are load-bearing and enforced by [import-linter](https://import-linter.readthedocs.io) contracts in [pyproject.toml](https://github.com/tsc-tools/aitana/blob/main/pyproject.toml):

1.  **Dependencies point downward only.** Lower layers never import higher ones.
2.  **Computation packages never fetch.** `features` operates on in-memory waveforms and must not import `access`. They are siblings that never import each other.

<!-- -->

                                     aitana-pipelines runs Snakemake workflows
                                            ┆
                                            ▼
                                     aitana-anomaly     detection
                                         │    │
                                         ▼    ▼
                           aitana-access   aitana-features   siblings: I/O ── computation
                                         │    |
                                         ▼    ▼
                                     aitana-storage     persistence

(Solid arrows are `depends on`. `aitana-pipelines` is where components are combined into reproducible pipelines: its Snakemake workflows *run* the lower layers in the active environment but `aitana.pipelines` never imports them (it depends on `aitana-volcanoes` and `aitana-storage` only). Grafana reads workflow results from `aitana-storage` through `aitana-api`. `aitana-bot` is a stub so far and imports no aitana package.)

`aitana-testing` provides synthetic waveforms for the package tests and is a dependency of `aitana-access`.

The `aitana-volcanoes` package is a convenience package that combines various datasets for a particular volcano.
