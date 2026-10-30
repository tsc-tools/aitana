# Implementation status


# `aitana-storage`

- [Storage](../reference/storage.storage.Storage.md#aitana.storage.storage.Storage) -- hierarchical feature archives (site/sensor/channel) backed by NetCDF or Zarr, with gap filling, incremental merging and time-range queries. Time stamps are snapped to each archive's grid on write and read, so jittered input and NetCDF's float-hour times come back exact.
- [AitanaDataset](../reference/storage.storage.AitanaDataset.md#aitana.storage.storage.AitanaDataset) -- PyTorch dataset yielding `[window_size, n_features]` windows with configurable stride; 2D features become one column per frequency, features must share time stamps, and an optional transform preprocesses the data. With `lazy=True` it reads from disk per batch; [SliceBatchSampler](../reference/storage.storage.SliceBatchSampler.md#aitana.storage.storage.SliceBatchSampler) makes each batch one contiguous read.
- `aitana-api` -- FastAPI service exposing `/feature` and `/inventory`.
- `grafana_annotations` -- sync the labels drawn on any Grafana dashboard with one label file; restores them to a freshly deployed Grafana.
- Background [IngestWorker](../reference/storage.ingest.IngestWorker.md#aitana.storage.ingest.IngestWorker) for queued dataset writes.


# `aitana-access`

- [tilde_request()](../reference/access.tilde.tilde_request.md#aitana.access.tilde.tilde_request) -- GeoNet Tilde v4 observations (lake temperature and level, gas, GNSS, water chemistry).
- [wfs_request()](../reference/access.wfs.wfs_request.md#aitana.access.wfs.wfs_request) -- GeoNet WFS earthquake catalogue by radius or polygon.
- [SeismicData](../reference/access.seismic.SeismicData.md#aitana.access.seismic.SeismicData) with pluggable clients tried in order: [S3Waveforms](../reference/access.seismic.S3Waveforms.md#aitana.access.seismic.S3Waveforms) (GeoNet's anonymous AWS Open Data bucket), [FDSNWaveforms](../reference/access.seismic.FDSNWaveforms.md#aitana.access.seismic.FDSNWaveforms), [SDSWaveforms](../reference/access.seismic.SDSWaveforms.md#aitana.access.seismic.SDSWaveforms), [MockSDSWaveforms](../reference/access.seismic.MockSDSWaveforms.md#aitana.access.seismic.MockSDSWaveforms); includes validation, gap filling and instrument-response removal.
- [cache_dataframe()](../reference/access.util.cache_dataframe.md#aitana.access.util.cache_dataframe) -- transparent disk caching with incremental date-range updates.
- [eqRate()](../reference/access.util.eqRate.md#aitana.access.util.eqRate) / [gradient()](../reference/access.util.gradient.md#aitana.access.util.gradient) -- moving-window earthquake rate and trend estimation.
- [AitanaAPI](../reference/access.api.AitanaAPI.md#aitana.access.api.AitanaAPI) -- FastAPI service adding `/waveform` and `/spectrogram`.


# `aitana-features`

RSAM (with wideband energy ratio), SSAM (filterbank and sonogram), DSAR, spectral features (predominant/central frequency, bandwidth, cepstral coefficients), complex-trace attributes (instantaneous frequency and bandwidth, normalised envelope, centroids), plus windowing and stacking utilities and Matplotlib/Plotly visualisation.


# `aitana-volcanoes`

- **Ruapehu** -- [CraterLake](../reference/volcanoes.ruapehu.CraterLake.md#aitana.volcanoes.ruapehu.CraterLake) (temperature, level), [Gas](../reference/volcanoes.ruapehu.Gas.md#aitana.volcanoes.ruapehu.Gas) (SO₂, CO₂, H₂S), [Seismicity](../reference/volcanoes.ruapehu.Seismicity.md#aitana.volcanoes.ruapehu.Seismicity) (regional and cone), [eruptions()](../reference/volcanoes.ruapehu.eruptions.md#aitana.volcanoes.ruapehu.eruptions), [load_all()](../reference/volcanoes.ruapehu.load_all.md#aitana.volcanoes.ruapehu.load_all), lake-height scaling.
- **Whakaari** -- [Gas](../reference/volcanoes.ruapehu.Gas.md#aitana.volcanoes.ruapehu.Gas), [Seismicity](../reference/volcanoes.ruapehu.Seismicity.md#aitana.volcanoes.ruapehu.Seismicity) (RSAM, quakes), [eruptions()](../reference/volcanoes.ruapehu.eruptions.md#aitana.volcanoes.ruapehu.eruptions), [load_all()](../reference/volcanoes.ruapehu.load_all.md#aitana.volcanoes.ruapehu.load_all).
- `assimilate` -- state-space Kalman models: [SO2FusionModel](../reference/volcanoes.assimilate.SO2FusionModel.md#aitana.volcanoes.assimilate.SO2FusionModel) (multi-sensor SO₂ fusion), [LocalLinearTrend](../reference/volcanoes.assimilate.LocalLinearTrend.md#aitana.volcanoes.assimilate.LocalLinearTrend), [SemiLinearTrend](../reference/volcanoes.assimilate.SemiLinearTrend.md#aitana.volcanoes.assimilate.SemiLinearTrend).


# `aitana-pipelines`

- `aitana-pipelines` CLI (`list` / [run](../reference/pipelines.cli.run.md#aitana.pipelines.cli.run) / [clean](../reference/pipelines.cli.clean.md#aitana.pipelines.cli.clean)) driving Snakemake workflows discovered via the `aitana.pipelines.workflows` entry-point group -- third-party packages can register their own.
- `scoring` -- pre-eruption-window evaluation, TP/FP/TN/FN thresholding, ROC curves, forecasted-rate computation.
- `visualise` -- [trellis_plot](../reference/pipelines.visualise.trellis_plot.md#aitana.pipelines.visualise.trellis_plot) (multi-panel time series with ensemble/uncertainty overlays) and [scoring_plot](../reference/pipelines.visualise.scoring_plot.md#aitana.pipelines.visualise.scoring_plot).
- Three bundled workflows: `egu2026`, a Bayesian network vs decision tree benchmark for Whakaari, `ruapehu_mp`, the Ruapehu RSAM matrix-profile alert (see [Running workflows](../user-guide/running-workflows.md#the-ruapehu-rsam-alert-end-to-end)), and `ruapehu_snet`, which stores scattering-network coefficients of Ruapehu's three-component data.


# `aitana-anomaly`

- [AnomalyDetectionBaseClass](../reference/anomaly.anomaly_base.AnomalyDetectionBaseClass.md#aitana.anomaly.anomaly_base.AnomalyDetectionBaseClass) -- the shared contract every detector implements, including [normalize_score()](../reference/anomaly.anomaly_base.AnomalyDetectionBaseClass.md#aitana.anomaly.anomaly_base.AnomalyDetectionBaseClass.normalize_score), the ECDF/percentile-rank normaliser that makes different algorithms' raw statistics comparable against a quiet reference period.
- [MinMaxScale](../reference/anomaly.anomaly_base.MinMaxScale.md#aitana.anomaly.anomaly_base.MinMaxScale) -- [AitanaDataset](../reference/storage.storage.AitanaDataset.md#aitana.storage.storage.AitanaDataset) transform that fills gaps with the median and scales each feature column to \[0, 1\] for model input.
- [AutoEncoder](../reference/anomaly.autoencoder.AutoEncoder.md#aitana.anomaly.autoencoder.AutoEncoder) / [TiedAutoEncoder](../reference/anomaly.autoencoder.TiedAutoEncoder.md#aitana.anomaly.autoencoder.TiedAutoEncoder) -- working per-timestep autoencoder with training, checkpointing and inference on a lazy [AitanaDataset](../reference/storage.storage.AitanaDataset.md#aitana.storage.storage.AitanaDataset); [FuzzyCluster](../reference/anomaly.autoencoder.FuzzyCluster.md#aitana.anomaly.autoencoder.FuzzyCluster) for cluster assignment. It needs the `pytorch` extra (`aitana-anomaly[pytorch]`).
- [SNet](../reference/anomaly.snet.SNet.md#aitana.anomaly.snet.SNet) -- scattering-network feature extraction (via `scatseisnet`): [preprocess()](../reference/anomaly.snet.SNet.md#aitana.anomaly.snet.SNet.preprocess) puts three components on a common 25 Hz grid with an anti-alias filter, [compute()](../reference/features.rsam.RSAM.md#aitana.features.rsam.RSAM.compute) gives first- and second-order coefficients of 20-min windows every 10 min, NaN where more than 10 % of a component is missing. The `ruapehu_snet` workflow runs it.
- [MatrixProfile](../reference/anomaly.matrix_profile.MatrixProfile.md#aitana.anomaly.matrix_profile.MatrixProfile) -- z-normalized left matrix profile (`stumpy`) of a single series, scored against its own trailing history and mapped to `[0, 1]` with [normalize_score()](../reference/anomaly.anomaly_base.AnomalyDetectionBaseClass.md#aitana.anomaly.anomaly_base.AnomalyDetectionBaseClass.normalize_score). Sensitive to changes in *shape*, deliberately blind to amplitude.
- `evaluation` -- the synthetic validation harness: [inject_faults()](../reference/anomaly.evaluation.inject_faults.md#aitana.anomaly.evaluation.inject_faults) (independent per-stream Bernoulli faults and coupled multi-stream faults), [inject_change_points()](../reference/anomaly.evaluation.inject_change_points.md#aitana.anomaly.evaluation.inject_change_points) (persistent regime shifts), [simulate_regime_series()](../reference/anomaly.evaluation.simulate_regime_series.md#aitana.anomaly.evaluation.simulate_regime_series) (synthetic data with known states) and [score_detector()](../reference/anomaly.evaluation.score_detector.md#aitana.anomaly.evaluation.score_detector) (ROC/AUC against the injected ground-truth mask). Documented with worked examples in [Evaluating anomaly detectors](../user-guide/evaluating-detectors.md).

**Not yet working in `aitana-anomaly`:** [MatrixProfile](../reference/anomaly.matrix_profile.MatrixProfile.md#aitana.anomaly.matrix_profile.MatrixProfile) is the only detector that implements [compute_anomaly_index()](../reference/anomaly.matrix_profile.MatrixProfile.md#aitana.anomaly.matrix_profile.MatrixProfile.compute_anomaly_index). `pca.py` needs a rewrite (its test suite is entirely skipped), and [SNet](../reference/anomaly.snet.SNet.md#aitana.anomaly.snet.SNet) has no downstream scoring layer.
