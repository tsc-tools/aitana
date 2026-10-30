# API Reference


## aitana.access


Pure I/O access layer for volcano monitoring data (GeoNet TILDE, WFS, seismic waveforms).


[access.tilde.tilde_request()](access.tilde.tilde_request.md#aitana.access.tilde.tilde_request)  
Request data from the tilde API (https://tilde.geonet.org.nz/v3/api-docs/).

[access.wfs.wfs_request()](access.wfs.wfs_request.md#aitana.access.wfs.wfs_request)  
Load volcano-tectonic earthquake catalogues from GeoNet's WFS service (http://wfs.geonet.org.nz).

[access.seismic.SeismicData](access.seismic.SeismicData.md#aitana.access.seismic.SeismicData)  

[access.seismic.WaveformBaseclass](access.seismic.WaveformBaseclass.md#aitana.access.seismic.WaveformBaseclass)  
Base class for waveform clients.

[access.seismic.SDSWaveforms](access.seismic.SDSWaveforms.md#aitana.access.seismic.SDSWaveforms)  

[access.seismic.FDSNWaveforms](access.seismic.FDSNWaveforms.md#aitana.access.seismic.FDSNWaveforms)  

[access.seismic.S3Waveforms](access.seismic.S3Waveforms.md#aitana.access.seismic.S3Waveforms)  

[access.seismic.MockSDSWaveforms](access.seismic.MockSDSWaveforms.md#aitana.access.seismic.MockSDSWaveforms)  
Mock SDSWaveforms class for testing by creating

[access.seismic.PostProcess](access.seismic.PostProcess.md#aitana.access.seismic.PostProcess)  
Class for checking waveforms and station metadata for valid values,

[access.seismic.PostProcessException](access.seismic.PostProcessException.md#aitana.access.seismic.PostProcessException)  

[access.seismic.get_instrument_response()](access.seismic.get_instrument_response.md#aitana.access.seismic.get_instrument_response)  
Retrieve instrument response file in STATIONXML format.

[access.api.AitanaAPI](access.api.AitanaAPI.md#aitana.access.api.AitanaAPI)  

[access.util.get_defaults_with_names()](access.util.get_defaults_with_names.md#aitana.access.util.get_defaults_with_names)  
Get parameter names and their default values

[access.util.generate_cache_key()](access.util.generate_cache_key.md#aitana.access.util.generate_cache_key)  
Generate a unique filename for caching based on function name and arguments.

[access.util.cache_dataframe()](access.util.cache_dataframe.md#aitana.access.util.cache_dataframe)  
Decorator to cache pandas DataFrames, handle date ranges, and persist the cache to disk.

[access.util.gradient()](access.util.gradient.md#aitana.access.util.gradient)  
Compute gradient for time series by first smoothing

[access.util.eqRate()](access.util.eqRate.md#aitana.access.util.eqRate)  
Compute earthquake rate.

[access.util.reindex()](access.util.reindex.md#aitana.access.util.reindex)  
Reindex and forward fill to generate a

[access.util.hex_to_rgb()](access.util.hex_to_rgb.md#aitana.access.util.hex_to_rgb)  
Return (red, green, blue) for the color given as \#rrggbb.

[access.util.rgb_to_hex()](access.util.rgb_to_hex.md#aitana.access.util.rgb_to_hex)  
Return color as \#rrggbb for the given color values.

[access.util.get_color()](access.util.get_color.md#aitana.access.util.get_color)  
Return a color from the matplotlib color cycle by index.

[access.logging_config.setup_logging()](access.logging_config.setup_logging.md#aitana.access.logging_config.setup_logging)  
Configure logging for the aitana package.

[access.logging_config.get_logger()](access.logging_config.get_logger.md#aitana.access.logging_config.get_logger)  
Get a logger for a specific module within aitana.


## aitana.volcanoes


Per-volcano domain aggregates for Ruapehu, Whakaari, and Taupo.


[volcanoes.whakaari.Gas](volcanoes.whakaari.Gas.md#aitana.volcanoes.whakaari.Gas)  

[volcanoes.whakaari.Seismicity](volcanoes.whakaari.Seismicity.md#aitana.volcanoes.whakaari.Seismicity)  

[volcanoes.whakaari.eruptions()](volcanoes.whakaari.eruptions.md#aitana.volcanoes.whakaari.eruptions)  
This function loads the eruption catalogue for White Island and

[volcanoes.whakaari.load_all()](volcanoes.whakaari.load_all.md#aitana.volcanoes.whakaari.load_all)  

[volcanoes.ruapehu.CraterLake](volcanoes.ruapehu.CraterLake.md#aitana.volcanoes.ruapehu.CraterLake)  

[volcanoes.ruapehu.Gas](volcanoes.ruapehu.Gas.md#aitana.volcanoes.ruapehu.Gas)  

[volcanoes.ruapehu.Seismicity](volcanoes.ruapehu.Seismicity.md#aitana.volcanoes.ruapehu.Seismicity)  

[volcanoes.ruapehu.eruptions()](volcanoes.ruapehu.eruptions.md#aitana.volcanoes.ruapehu.eruptions)  
This function loads the eruption catalogue for Mt. Ruapehu and declusters

[volcanoes.ruapehu.load_all()](volcanoes.ruapehu.load_all.md#aitana.volcanoes.ruapehu.load_all)  

[volcanoes.ruapehu.lake_height_scaling()](volcanoes.ruapehu.lake_height_scaling.md#aitana.volcanoes.ruapehu.lake_height_scaling)  
Calculate the scaling factor between the lake height sensors 08 and 03.

[volcanoes.assimilate.SemiLinearTrend](volcanoes.assimilate.SemiLinearTrend.md#aitana.volcanoes.assimilate.SemiLinearTrend)  

[volcanoes.assimilate.LocalLinearTrend](volcanoes.assimilate.LocalLinearTrend.md#aitana.volcanoes.assimilate.LocalLinearTrend)  

[volcanoes.assimilate.SO2FusionModel](volcanoes.assimilate.SO2FusionModel.md#aitana.volcanoes.assimilate.SO2FusionModel)  

[volcanoes.get_data()](volcanoes.get_data.md#aitana.volcanoes.get_data)  
Get path to bundled data in the aitana.volcanoes package.


## aitana.features


Algorithms over in-memory waveforms for volcano monitoring (RSAM, DSAR, SSAM).


[features.feature_base.FeatureBaseClass](features.feature_base.FeatureBaseClass.md#aitana.features.feature_base.FeatureBaseClass)  

[features.rsam.RSAM](features.rsam.RSAM.md#aitana.features.rsam.RSAM)  
Compute RSAM as the mean of the absolute value of a signal filtered

[features.rsam.rsam_wideband_energy_ratio()](features.rsam.rsam_wideband_energy_ratio.md#aitana.features.rsam.rsam_wideband_energy_ratio)  

[features.dsar.DSAR](features.dsar.DSAR.md#aitana.features.dsar.DSAR)  
DSAR is the ratio of

[features.ssam.SSAM](features.ssam.SSAM.md#aitana.features.ssam.SSAM)  
SSAM is basically the same as computing a spectrogram. Data are split into

[features.spectral_features.SpectralFeatures](features.spectral_features.SpectralFeatures.md#aitana.features.spectral_features.SpectralFeatures)  

[features.cpxtrace.Inst_Band](features.cpxtrace.Inst_Band.md#aitana.features.cpxtrace.Inst_Band)  
Computes the instantaneous bandwidth of the given data which can be

[features.cpxtrace.Inst_Freq](features.cpxtrace.Inst_Freq.md#aitana.features.cpxtrace.Inst_Freq)  
Computes the instantaneous frequency of the given data which can be

[features.cpxtrace.Norm_Env](features.cpxtrace.Norm_Env.md#aitana.features.cpxtrace.Norm_Env)  
Computes the normalized envelope of the given data which can be

[features.cpxtrace.Centroid](features.cpxtrace.Centroid.md#aitana.features.cpxtrace.Centroid)  
Computes the centroid time of the given data which can be windowed or

[features.cpxtrace.WindowCentroid](features.cpxtrace.WindowCentroid.md#aitana.features.cpxtrace.WindowCentroid)  
Computes the centroid of a given time window,

[features.util.stride_windows()](features.util.stride_windows.md#aitana.features.util.stride_windows)  
Get all windows of x with length n as a single array,

[features.util.demean()](features.util.demean.md#aitana.features.util.demean)  
Return x minus the mean(x).

[features.util.apply_hanning()](features.util.apply_hanning.md#aitana.features.util.apply_hanning)  
Apply a hanning window to the given 1D or 2D array along

[features.util.window_array()](features.util.window_array.md#aitana.features.util.window_array)  
Take a 1-D numpy array and devide it into (overlapping)

[features.util.trace_window_times()](features.util.trace_window_times.md#aitana.features.util.trace_window_times)  
Subdivide an `obspy.core.trace.Trace` into windows of `window_len`

[features.util.trace_window_data()](features.util.trace_window_data.md#aitana.features.util.trace_window_data)  
Subdivide an `obspy.core.trace.Trace` into windows of `interval`

[features.util.apply_freq_filter()](features.util.apply_freq_filter.md#aitana.features.util.apply_freq_filter)  
Apply frequency filter to seismic data.

[features.util.round_time()](features.util.round_time.md#aitana.features.util.round_time)  
Find closest multiple of interval to time.

[features.util.stack_feature()](features.util.stack_feature.md#aitana.features.util.stack_feature)  
Stack features in a dataarray.

[features.visualise.spectrogram_preprocessing()](features.visualise.spectrogram_preprocessing.md#aitana.features.visualise.spectrogram_preprocessing)  

[features.visualise.plot_ssam_mpl()](features.visualise.plot_ssam_mpl.md#aitana.features.visualise.plot_ssam_mpl)  
Plot spectrogram data using matplotlib.

[features.visualise.multi_feature_plot_mpl()](features.visualise.multi_feature_plot_mpl.md#aitana.features.visualise.multi_feature_plot_mpl)  
Plot two features in one figure. 2D features have to be plotted first.

[features.visualise.plot_ssam_plotly()](features.visualise.plot_ssam_plotly.md#aitana.features.visualise.plot_ssam_plotly)  
Plot spectrogram data using matplotlib.

[features.visualise.get_trace()](features.visualise.get_trace.md#aitana.features.visualise.get_trace)  
Return a plotly graphics object from a feature.

[features.visualise.multi_feature_plot()](features.visualise.multi_feature_plot.md#aitana.features.visualise.multi_feature_plot)  
Return a combined time-series plot of two features.

[features.get_data()](features.get_data.md#aitana.features.get_data)  
Return path to aitana.features package.


## aitana.anomaly


Anomaly detection in time-series data.


[anomaly.anomaly_base.AnomalyDetectionBaseClass](anomaly.anomaly_base.AnomalyDetectionBaseClass.md#aitana.anomaly.anomaly_base.AnomalyDetectionBaseClass)  

[anomaly.anomaly_base.min_max_scale_netcdf()](anomaly.anomaly_base.min_max_scale_netcdf.md#aitana.anomaly.anomaly_base.min_max_scale_netcdf)  

[anomaly.pca.PCA](anomaly.pca.PCA.md#aitana.anomaly.pca.PCA)  

[anomaly.pca.runPca()](anomaly.pca.runPca.md#aitana.anomaly.pca.runPca)  

[anomaly.pca.runPcaCore()](anomaly.pca.runPcaCore.md#aitana.anomaly.pca.runPcaCore)  

[anomaly.pca.ModelException](anomaly.pca.ModelException.md#aitana.anomaly.pca.ModelException)  

[anomaly.matrix_profile.MatrixProfile](anomaly.matrix_profile.MatrixProfile.md#aitana.anomaly.matrix_profile.MatrixProfile)  
Score how unlike its own past each subsequence of a time series is.

[anomaly.snet.SNet](anomaly.snet.SNet.md#aitana.anomaly.snet.SNet)  

[anomaly.autoencoder.AutoEncoder](anomaly.autoencoder.AutoEncoder.md#aitana.anomaly.autoencoder.AutoEncoder)  

[anomaly.autoencoder.TiedAutoEncoder](anomaly.autoencoder.TiedAutoEncoder.md#aitana.anomaly.autoencoder.TiedAutoEncoder)  
Applies an tied-weight autoencoder to the incoming data.

[anomaly.autoencoder.FuzzyCluster](anomaly.autoencoder.FuzzyCluster.md#aitana.anomaly.autoencoder.FuzzyCluster)  

[anomaly.anomaly_base.MinMaxScale](anomaly.anomaly_base.MinMaxScale.md#aitana.anomaly.anomaly_base.MinMaxScale)  
Fill gaps with the median and scale each feature column to \[0, 1\].

[anomaly.evaluation.inject_faults()](anomaly.evaluation.inject_faults.md#aitana.anomaly.evaluation.inject_faults)  
Inject synthetic faults into a DataFrame for anomaly detection evaluation.

[anomaly.evaluation.inject_change_points()](anomaly.evaluation.inject_change_points.md#aitana.anomaly.evaluation.inject_change_points)  
Inject persistent regime shifts (change points) into a DataFrame.

[anomaly.evaluation.simulate_regime_series()](anomaly.evaluation.simulate_regime_series.md#aitana.anomaly.evaluation.simulate_regime_series)  
Generate a synthetic multivariate time series stitched from discrete regimes.

[anomaly.evaluation.quiet_series()](anomaly.evaluation.quiet_series.md#aitana.anomaly.evaluation.quiet_series)  
Generate a quiet background: a repeating template plus small noise.

[anomaly.evaluation.score_detector()](anomaly.evaluation.score_detector.md#aitana.anomaly.evaluation.score_detector)  
Score a detector against the fault-injection ground-truth mask.


## aitana.storage


Store time series data as HDF5 (netCDF) or Zarr files and access them through an API.


[storage.storage.Storage](storage.storage.Storage.md#aitana.storage.storage.Storage)  
Query computed features

[storage.storage.Path](storage.storage.Path.md#aitana.storage.storage.Path)  

[storage.storage.AitanaDataset](storage.storage.AitanaDataset.md#aitana.storage.storage.AitanaDataset)  
PyTorch Dataset of features from a store.

[storage.storage.SliceBatchSampler](storage.storage.SliceBatchSampler.md#aitana.storage.storage.SliceBatchSampler)  
Batch sampler that yields slices of consecutive sample indices.

[storage.ingest.IngestWorker](storage.ingest.IngestWorker.md#aitana.storage.ingest.IngestWorker)  

[storage.ingest.enqueue_dataset()](storage.ingest.enqueue_dataset.md#aitana.storage.ingest.enqueue_dataset)  
Enqueue a dataset for ingestion.

[storage.api.StorageAPI](storage.api.StorageAPI.md#aitana.storage.api.StorageAPI)  

[storage.api.parse_path()](storage.api.parse_path.md#aitana.storage.api.parse_path)  
Split a comma-separated path string into (group, subdir_list).

[storage.xarray2netcdf.xarray2netcdf()](storage.xarray2netcdf.xarray2netcdf.md#aitana.storage.xarray2netcdf.xarray2netcdf)  
Store an xarray dataset as an HDF5 file.

[storage.xarray2zarr.xarray2zarr()](storage.xarray2zarr.xarray2zarr.md#aitana.storage.xarray2zarr.xarray2zarr)  
Write xarray dataset to zarr files.

[storage.xarray2zarr.PathNotFoundError](storage.xarray2zarr.PathNotFoundError.md#aitana.storage.xarray2zarr.PathNotFoundError)  

[storage.grafana_annotations.sync_labels()](storage.grafana_annotations.sync_labels.md#aitana.storage.grafana_annotations.sync_labels)  
Bring the label annotations in Grafana and the stored labels in step.

[storage.grafana_annotations.is_label()](storage.grafana_annotations.is_label.md#aitana.storage.grafana_annotations.is_label)  
Whether tags mark an annotation as a `label:` label.

[storage.utils.generate_test_data()](storage.utils.generate_test_data.md#aitana.storage.utils.generate_test_data)  
Generate a 1D or 2D feature for testing.

[storage.utils.round_datetime()](storage.utils.round_datetime.md#aitana.storage.utils.round_datetime)  
Find closest multiple of interval to given time.

[storage.utils.floor_datetime()](storage.utils.floor_datetime.md#aitana.storage.utils.floor_datetime)  
Floor a datetime to the latest multiple of a given interval.

[storage.utils.get_dt()](storage.utils.get_dt.md#aitana.storage.utils.get_dt)  
Infer the sampling interval of a time axis.

[storage.utils.snap_to_grid()](storage.utils.snap_to_grid.md#aitana.storage.utils.snap_to_grid)  
Round time stamps to the nearest point of a regular grid.

[storage.utils.fill_time_gaps()](storage.utils.fill_time_gaps.md#aitana.storage.utils.fill_time_gaps)  
Fill gaps in time series with NaN values by reindexing to a complete datetime range.

[storage.utils.merge_arrays()](storage.utils.merge_arrays.md#aitana.storage.utils.merge_arrays)  
Merge two xarray datasets with the same datetime index.

[storage.get_data()](storage.get_data.md#aitana.storage.get_data)  
Return path to aitana.storage package.


## aitana.pipelines


Reproducible workflows that run and evaluate aitana components for volcano monitoring.


[pipelines.cli.WorkflowDescriptor](pipelines.cli.WorkflowDescriptor.md#aitana.pipelines.cli.WorkflowDescriptor)  

[pipelines.cli.PipelineBackend](pipelines.cli.PipelineBackend.md#aitana.pipelines.cli.PipelineBackend)  

[pipelines.cli.SnakemakeBackend](pipelines.cli.SnakemakeBackend.md#aitana.pipelines.cli.SnakemakeBackend)  

[pipelines.cli.PipelineError](pipelines.cli.PipelineError.md#aitana.pipelines.cli.PipelineError)  

[pipelines.cli.discover_workflows()](pipelines.cli.discover_workflows.md#aitana.pipelines.cli.discover_workflows)  

[pipelines.cli.run()](pipelines.cli.run.md#aitana.pipelines.cli.run)  
Run the workflow registered for a volcano.

[pipelines.cli.clean()](pipelines.cli.clean.md#aitana.pipelines.cli.clean)  
Delete the outputs of a volcano's workflow, chosen with `--workflow` if it has several.

[pipelines.cli.list_workflows()](pipelines.cli.list_workflows.md#aitana.pipelines.cli.list_workflows)  
List all registered workflows.

[pipelines.scoring.get_evaluation_windows()](pipelines.scoring.get_evaluation_windows.md#aitana.pipelines.scoring.get_evaluation_windows)  
Get the evaluation windows for the forecasted rates.

[pipelines.scoring.compute_rates()](pipelines.scoring.compute_rates.md#aitana.pipelines.scoring.compute_rates)  
Compute the forecasted rates for positive and negative windows.

[pipelines.scoring.evaluate_threshold()](pipelines.scoring.evaluate_threshold.md#aitana.pipelines.scoring.evaluate_threshold)  
Evaluate the number of true positives, false positives, true negatives and false negatives for a given threshold.

[pipelines.scoring.get_roc_curve()](pipelines.scoring.get_roc_curve.md#aitana.pipelines.scoring.get_roc_curve)  
Compute the ROC curve for a given model and thresholds.

[pipelines.scoring.make_strictly_increasing()](pipelines.scoring.make_strictly_increasing.md#aitana.pipelines.scoring.make_strictly_increasing)  
Make a sequence strictly increasing. Some of the ROC curves computed with

[pipelines.visualise.set_style()](pipelines.visualise.set_style.md#aitana.pipelines.visualise.set_style)  

[pipelines.visualise.trellis_plot()](pipelines.visualise.trellis_plot.md#aitana.pipelines.visualise.trellis_plot)  
Create a trellis plot for the given models and data.

[pipelines.visualise.scoring_plot()](pipelines.visualise.scoring_plot.md#aitana.pipelines.visualise.scoring_plot)  
Plot the forecast probabilities and the evaluation windows.

[pipelines.get_data()](pipelines.get_data.md#aitana.pipelines.get_data)  
Get path to bundled data in the aitana.pipelines package (workflow files, etc.).


## aitana.testing


Synthetic seismic waveforms with known spectral content for testing aitana packages.


[testing.synthetic.synthetic_trace()](testing.synthetic.synthetic_trace.md#aitana.testing.synthetic.synthetic_trace)  
Produce a test signal for which we know where the peaks
