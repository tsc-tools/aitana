"""
Compute 10-min RSAM for one time span (a UTC day, or the current partial day).
"""

import warnings

import numpy as np
import pandas as pd
from obspy import UTCDateTime

from aitana.access.seismic import FDSNWaveforms, S3Waveforms, SeismicData
from aitana.features.rsam import RSAM

INTERVAL = 600  # seconds
MAX_GAP_FRACTION = 0.1  # windows missing more data than this are NaN

net, sta, loc, cha = snakemake.config["seed_id"].split(".")
start = UTCDateTime(snakemake.params.start)
end = UTCDateTime(snakemake.params.end)

# The S3 archive lags about a day behind; the near-real-time FDSN service
# covers the most recent days.
sd = SeismicData([S3Waveforms(), FDSNWaveforms("https://service-nrt.geonet.org.nz")])
tr = next(sd.get_waveforms(net, sta, loc, cha, start, end))

rsam = pd.Series(dtype=float, name="rsam")
if len(tr) > 0 and not np.all(np.isnan(tr.data)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rsam = (
            RSAM(interval=INTERVAL, max_gap_fraction=MAX_GAP_FRACTION)
            .compute(tr)["rsam"]
            .to_series()
        )
    # Traces can start a few milliseconds after the requested time.
    rsam.index = rsam.index.round(f"{INTERVAL}s")

rsam.rename_axis("datetime").to_csv(snakemake.output[0])
