"""
Scattering coefficients for one time span (a UTC day plus the 10 min its last
window needs, or the current partial day).
"""

import warnings

import xarray as xr
from obspy import UTCDateTime

from aitana.access.seismic import FDSNWaveforms, S3Waveforms, SeismicData
from aitana.anomaly.snet import SNet

net, sta, loc, cha = snakemake.config["seed_id"].split(".")
start = UTCDateTime(snakemake.params.start)
end = UTCDateTime(snakemake.params.end)

# The S3 archive lags about a day behind; the near-real-time FDSN service
# covers the most recent days.
sd = SeismicData([S3Waveforms(), FDSNWaveforms("https://service-nrt.geonet.org.nz")])
st = next(sd.get_waveforms(net, sta, loc, cha, start, end))

snet = SNet()
if len(st) == 3:
    feature = snet.compute(snet.preprocess(st, start, end))
else:
    warnings.warn(f"{len(st)} of 3 components from {start} to {end}, not computed.")
    feature = xr.Dataset()
feature.to_netcdf(snakemake.output[0])
