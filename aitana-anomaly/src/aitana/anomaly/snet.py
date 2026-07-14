"""
Compute scattering transform and UMAP for seismic data. To run as a standalone script, tracking the experiment with mlflow use:
```
snet --starttime 2022-01-01T00:00:00 --endtime 2022-01-02T00:00:00 --data_dir /path/to/data --experiment snet_experiment \
    --mlflow_tracking_uri http://localhost:5000 --mlflow_run_name snet_run --stream NET.STA.LOC.CHAN
```
"""
import logging
from obspy import UTCDateTime
from aitana.access.seismic import SeismicData as DataSource, S3Waveforms
from aitana.features.util import window_array
from scatseisnet import ScatteringNetwork
import numpy as np
from obspy import Stream
import xarray as xr
import pandas as pd
from aitana.storage import Storage
from aitana.features.feature_base import FeatureBaseClass
from tqdm import tqdm  # progress bar

logger = logging.getLogger(__name__)

# New helper to configure logging
def configure_logging(log_file: str, level: int = logging.INFO):
    """Configure logging to file and limit console output.

    Parameters
    ----------
    log_file : str
        Path to the log file where detailed logs will be written.
    level : int
        Logging level for the file handler.
    """
    root = logging.getLogger()
    # Avoid duplicate handlers if reconfigured
    if getattr(root, "_snet_logging_configured", False):
        return
    root.setLevel(logging.DEBUG)

    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)

    # Console handler (only errors so progress bar remains clean)
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    root.handlers.clear()
    root.addHandler(fh)
    root.addHandler(ch)
    root._snet_logging_configured = True
    logger.debug("Logging configured. File: %s", log_file)

class SNet(FeatureBaseClass):
    def __init__(self):
        super(SNet, self).__init__()
        self.segment_duration_seconds = 1200.0
        self.sampling_rate_hertz = 25.0
        self.overlap = 0.5
        self.steplength = int(self.segment_duration_seconds*self.sampling_rate_hertz) - int(self.segment_duration_seconds*self.sampling_rate_hertz * self.overlap)
        self.samples_per_segment = int(self.segment_duration_seconds * self.sampling_rate_hertz)
        self.bank_keyword_arguments = (
            {"octaves": 6, "resolution": 4, "quality": 2},
            {"octaves": 6, "resolution": 1, "quality": 2},
        )
        msg = f"SNet initialized with segment_duration_seconds={self.segment_duration_seconds}, "
        msg += f"sampling_rate_hertz={self.sampling_rate_hertz}, overlap={self.overlap}, "
        msg += f"steplength={self.steplength}, samples_per_segment={self.samples_per_segment}" 
        logger.debug(msg)

    def compute(self, st: Stream):
        logger.debug("Computing scattering coefficients.")
        window_len = int(self.segment_duration_seconds * self.sampling_rate_hertz)
        noverlap = int(self.segment_duration_seconds * self.sampling_rate_hertz * self.overlap)
        resultZ, windowValsZ = window_array(st[0].data, window_len, noverlap)
        result1, windowVals1 = window_array(st[1].data, window_len, noverlap)
        result2, windowVals2 = window_array(st[2].data, window_len, noverlap)
        segments = np.stack([resultZ.T, result1.T, result2.T], axis=1)
        timestamps = [(st[0].stats.starttime + _t).datetime for _t in np.arange(resultZ.shape[1])*self.steplength / self.sampling_rate_hertz]
        network = ScatteringNetwork(
            *self.bank_keyword_arguments,
            bins=self.samples_per_segment,
            sampling_rate=self.sampling_rate_hertz,
        )
        time_debug_start = UTCDateTime.now()
        scattering_coefficients = network.transform(segments, reduce_type=np.median)
        time_debug_end = UTCDateTime.now()
        logger.debug(f"Scattering coefficients computed in {time_debug_end - time_debug_start} seconds.")
        coords_order1 = {'datetime': pd.to_datetime(timestamps), 'channel': np.arange(scattering_coefficients[0].shape[1]),
                        'frequency': network.banks[0].centers}
        coords_order2 = coords_order1.copy()
        coords_order2['order2 frequency'] = network.banks[1].centers 
        xda_order1 = xr.DataArray(scattering_coefficients[0], dims=['datetime', 'channel', 'frequency'],
                                coords=coords_order1)
        xda_order2 = xr.DataArray(scattering_coefficients[1], dims=['datetime', 'channel', 'frequency', 'order2 frequency'],
                                coords=coords_order2)
        self.feature = xr.Dataset({'order1': xda_order1, 'order2': xda_order2})
        self.feature.attrs["starttime"] = st[0].stats.starttime.isoformat()
        self.feature.attrs["endtime"] = st[0].stats.endtime.isoformat()
        self.feature.attrs["station"] = st[0].stats.station
        return self.feature

def main(argv=None):
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from aitana.features.visualise import plot_ssam_mpl
    from aitana.storage.utils import floor_datetime
    try:
        import mlflow
    except ImportError:
        logger.warning("mlflow is not installed. MLflow logging will be disabled.")
        mlflow_log = False 
    parse = ArgumentParser(description=__doc__ or "",
                           formatter_class=RawDescriptionHelpFormatter)
    parse.add_argument('-s', '--starttime',
                      help="Define starttime in ISO860 format, e.g. 2022-01-01T00:00:00",
                      default="2022-01-01T00:00:00")
    parse.add_argument('-e', '--endtime',
                      help="Define endtime in ISO860 format, e.g. 2022-01-02T00:00:00",
                      default="2022-01-02T00:00:00")
    parse.add_argument('-d', '--data_dir',
                      help="Define data directory, e.g. /home/zizou/data/robust_forecasting",
                      default="/home/zizou/data/robust_forecasting")
    parse.add_argument('-c', '--chunks',
                      help="Define chunk size for Zarr storage.",
                      default=144, type=int) 
    parse.add_argument('--experiment',
                      help="Define MLflow experiment name.",
                      default="scatseisnet")
    parse.add_argument('--mlflow_tracking_uri',
                      help="Define MLflow tracking URI.",
                      default=None)
    parse.add_argument('--mlflow_run_name',
                      help="Define MLflow run name.",
                      default=None)
    parse.add_argument('--plot-only', \
                      help="Only plot the feature without computing it.", \
                      action='store_true')
    parse.add_argument('--log-file', \
                      help="Log file path.", \
                      default='snet-log.txt')
    parse.add_argument('--stream', type=str,
                       help="Seismic stream identifier in the format NET.STA.LOC.CHAN",
                       default="NZ.MAVZ.10.HH?")
    args = parse.parse_args(argv)

    # Configure logging
    configure_logging(args.log_file, logging.DEBUG)
    logger.info("Starting snet with args: %s", args)

    s3w = S3Waveforms(staxml_dir=args.data_dir) 
    ds = DataSource(clients=[s3w], chunk_size=86400)
    starttime = floor_datetime(UTCDateTime(args.starttime), 1200)
    endtime = floor_datetime(UTCDateTime(args.endtime), 1200)
    logger.debug(f"Starttime: {starttime}, Endtime: {endtime}")
    net, sta, loc, chan = args.stream.split(".")
    g = Storage(args.experiment, rootdir=args.data_dir, backend="netcdf",
                starttime=starttime.datetime, endtime=endtime.datetime)
    c = g.get_substore(sta)
    try:
        last_datapoint = c('order1', metadata=True).last_datapoint.values[-1]
        logger.debug(f"Last datapoint in storage: {last_datapoint}")
        starttime = UTCDateTime(max(starttime.datetime, pd.to_datetime(last_datapoint)))
        starttime = floor_datetime(starttime, 1200)
    except Exception as e:
        logger.warning(f"Could not update starttime from last datapoint: {e}")
    if not args.plot_only:
        logger.info(f"Fetching data from {starttime} to {endtime} for stream {args.stream}")
        # Progress bar over waveform chunks
        for st in tqdm(ds.get_waveforms(net, sta, loc, chan, starttime, endtime, cache=True),
                       desc="Waveform chunks", unit="chunk", ):
            if len(st) < 3:
                logger.warning(f"Stream has less than 3 components for {starttime} - {endtime}.")
                continue
            for tr in st:
                isn = np.isnan(tr.data)
                if isn.any(): 
                    idx = np.where(~isn, np.arange(len(tr.data)), 0)
                    if np.isnan(tr.data[0]):
                        tr.data[0] = tr.data[np.where(~isn)[0][0]]
                    np.maximum.accumulate(idx, out=idx)
                    tr.data[isn] = tr.data[idx[isn]]
                tr.detrend("linear")
                tr.interpolate(sampling_rate=25)
            s = SNet()
            feature = s.compute(st)
            feature['datetime'] = feature.datetime.dt.round("600s")
            logger.info(f"Computed feature with shape {feature['order1'].shape} and {feature['order2'].shape}")
            logger.info(f"Feature starttime: {feature.attrs['starttime']}, endtime: {feature.attrs['endtime']}")
            logger.info(f"Feature station: {feature.attrs['station']}")
            c.save(feature)
    xds = c('order1')
    fig = plot_ssam_mpl(xds.loc[dict(channel=0)].sel(frequency=slice(None, None, -1)).transpose('frequency', 'datetime'),
                        dbscale=True, canvas_dim=None, log=True)
    if mlflow_log:
        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
            experiment = mlflow.set_experiment(args.experiment)
            with mlflow.start_run(run_name=args.mlflow_run_name) as run:
                mlflow.log_param("starttime", str(starttime))
                mlflow.log_param("endtime", str(endtime))
                mlflow.log_figure(fig, artifact_file="plots/scattering_coefficients_plot.png")
                mlflow.log_artifact("snet-log.txt", artifact_path="logs")
    else:
        fig.savefig("scattering_coefficients_plot.png")

if __name__ == "__main__":
    main()
