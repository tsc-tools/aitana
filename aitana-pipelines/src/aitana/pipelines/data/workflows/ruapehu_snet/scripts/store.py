"""
Store the scattering coefficients of this run's days in aitana-storage and plot
the first-order coefficients of the vertical component.
"""

import pandas as pd
import xarray as xr
from aitana.features.visualise import plot_ssam_mpl
from aitana.storage import Storage

cfg = snakemake.config

frames = [xr.load_dataset(f) for f in snakemake.input]
frames = [f for f in frames if "order1" in f]
if not frames:
    raise ValueError("No three-component data in any day of this run.")
snet = xr.concat(frames, dim="datetime")
# Storage infers the resolution from the first two timestamps, so days without
# data become NaN rather than a hole in the time axis.
snet = snet.reindex(
    datetime=pd.date_range(snet.datetime.values[0], snet.datetime.values[-1], freq="10min")
)

net, sta, loc, cha = cfg["seed_id"].split(".")
store = Storage("ruapehu", cfg["storage_dir"]).get_substore(sta, loc, cha.replace("?", ""))
store.save(snet.rename(order1="snet_order1", order2="snet_order2"))

# channel 0 is Z, the first component SeismicData returns
order1 = snet["order1"].sel(channel=0)
fig = plot_ssam_mpl(
    order1.sel(frequency=slice(None, None, -1)).transpose("frequency", "datetime"),
    dbscale=True,
    canvas_dim=None,
    log=True,
)
fig.savefig(snakemake.output.plot)
