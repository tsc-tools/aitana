---
name: aitana
description: >
  Validation and benchmarking framework for AI/ML in volcano monitoring. Use when writing Python code that uses the aitana package.
---

# Aitana

Validation and benchmarking framework for AI/ML in volcano monitoring.

## Installation

```bash
pip install aitana
```

## API overview

### aitana.access

Pure I/O access layer for volcano monitoring data (GeoNet TILDE, WFS, seismic waveforms).

- `access.tilde.tilde_request`
- `access.wfs.wfs_request`
- `access.seismic.SeismicData`
- `access.seismic.WaveformBaseclass`
- `access.seismic.SDSWaveforms`
- `access.seismic.FDSNWaveforms`
- `access.seismic.S3Waveforms`
- `access.seismic.MockSDSWaveforms`
- `access.seismic.PostProcess`
- `access.seismic.PostProcessException`
- `access.seismic.get_instrument_response`
- `access.api.AitanaAPI`
- `access.util.get_defaults_with_names`
- `access.util.generate_cache_key`
- `access.util.cache_dataframe`
- `access.util.gradient`
- `access.util.eqRate`
- `access.util.reindex`
- `access.util.hex_to_rgb`
- `access.util.rgb_to_hex`
- `access.util.get_color`
- `access.logging_config.setup_logging`
- `access.logging_config.get_logger`

### aitana.volcanoes

Per-volcano domain aggregates for Ruapehu, Whakaari, and Taupo.

- `volcanoes.whakaari.Gas`
- `volcanoes.whakaari.Seismicity`
- `volcanoes.whakaari.eruptions`
- `volcanoes.whakaari.load_all`
- `volcanoes.ruapehu.CraterLake`
- `volcanoes.ruapehu.Gas`
- `volcanoes.ruapehu.Seismicity`
- `volcanoes.ruapehu.eruptions`
- `volcanoes.ruapehu.load_all`
- `volcanoes.ruapehu.lake_height_scaling`
- `volcanoes.assimilate.SemiLinearTrend`
- `volcanoes.assimilate.LocalLinearTrend`
- `volcanoes.assimilate.SO2FusionModel`
- `volcanoes.get_data`

### aitana.features

Algorithms over in-memory waveforms for volcano monitoring (RSAM, DSAR, SSAM).

- `features.feature_base.FeatureBaseClass`
- `features.rsam.RSAM`
- `features.rsam.rsam_wideband_energy_ratio`
- `features.dsar.DSAR`
- `features.ssam.SSAM`
- `features.spectral_features.SpectralFeatures`
- `features.cpxtrace.Inst_Band`
- `features.cpxtrace.Inst_Freq`
- `features.cpxtrace.Norm_Env`
- `features.cpxtrace.Centroid`
- `features.cpxtrace.WindowCentroid`
- `features.util.stride_windows`
- `features.util.demean`
- `features.util.apply_hanning`
- `features.util.window_array`
- `features.util.trace_window_times`
- `features.util.trace_window_data`
- `features.util.apply_freq_filter`
- `features.util.round_time`
- `features.util.stack_feature`
- `features.visualise.spectrogram_preprocessing`
- `features.visualise.plot_ssam_mpl`
- `features.visualise.multi_feature_plot_mpl`
- `features.visualise.plot_ssam_plotly`
- `features.visualise.get_trace`
- `features.visualise.multi_feature_plot`
- `features.get_data`

### aitana.anomaly

Anomaly detection in time-series data.

- `anomaly.anomaly_base.AnomalyDetectionBaseClass`
- `anomaly.anomaly_base.min_max_scale_netcdf`
- `anomaly.pca.PCA`
- `anomaly.pca.runPca`
- `anomaly.pca.runPcaCore`
- `anomaly.pca.ModelException`
- `anomaly.matrix_profile.MatrixProfile`
- `anomaly.snet.SNet`
- `anomaly.autoencoder.AutoEncoder`
- `anomaly.autoencoder.TiedAutoEncoder`
- `anomaly.autoencoder.FuzzyCluster`
- `anomaly.dataset.ZizouDataset`
- `anomaly.dataset.SliceBatchSampler`
- `anomaly.evaluation.inject_faults`
- `anomaly.evaluation.inject_change_points`
- `anomaly.evaluation.simulate_regime_series`
- `anomaly.evaluation.quiet_series`
- `anomaly.evaluation.score_detector`

### aitana.storage

Store time series data as HDF5 (netCDF) or Zarr files and access them through an API.

- `storage.storage.Storage`
- `storage.storage.Path`
- `storage.storage.AitanaDataset`
- `storage.ingest.IngestWorker`
- `storage.ingest.enqueue_dataset`
- `storage.api.StorageAPI`
- `storage.api.parse_path`
- `storage.xarray2netcdf.xarray2netcdf`
- `storage.xarray2zarr.xarray2zarr`
- `storage.xarray2zarr.PathNotFoundError`
- `storage.grafana_annotations.sync_labels`
- `storage.grafana_annotations.is_label`
- `storage.utils.generate_test_data`
- `storage.utils.round_datetime`
- `storage.utils.floor_datetime`
- `storage.utils.get_dt`
- `storage.utils.fill_time_gaps`
- `storage.utils.merge_arrays`
- `storage.get_data`

### aitana.pipelines

Reproducible workflows that run and evaluate aitana components for volcano monitoring.

- `pipelines.cli.WorkflowDescriptor`
- `pipelines.cli.PipelineBackend`
- `pipelines.cli.SnakemakeBackend`
- `pipelines.cli.PipelineError`
- `pipelines.cli.discover_workflows`
- `pipelines.cli.run`
- `pipelines.cli.clean`
- `pipelines.cli.list_workflows`
- `pipelines.scoring.get_evaluation_windows`
- `pipelines.scoring.compute_rates`
- `pipelines.scoring.evaluate_threshold`
- `pipelines.scoring.get_roc_curve`
- `pipelines.scoring.make_strictly_increasing`
- `pipelines.visualise.set_style`
- `pipelines.visualise.trellis_plot`
- `pipelines.visualise.scoring_plot`
- `pipelines.get_data`

### aitana.testing

Synthetic seismic waveforms with known spectral content for testing aitana packages.

- `testing.synthetic.synthetic_trace`

## Resources

- [Full documentation](https://tsc-tools.github.io/aitana/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs
