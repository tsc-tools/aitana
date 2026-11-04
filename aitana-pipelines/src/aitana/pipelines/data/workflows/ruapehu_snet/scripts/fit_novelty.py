"""
Fit the novelty detector of the scattering coefficients on the reference days.
"""

import joblib
import xarray as xr
from aitana.anomaly.snet import SNetNovelty

frames = [xr.load_dataset(f) for f in snakemake.input]
frames = [f for f in frames if "order1" in f]
if not frames:
    raise ValueError("No three-component data in any reference day.")
joblib.dump(SNetNovelty().fit(xr.concat(frames, dim="datetime")), snakemake.output[0])
