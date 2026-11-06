"""
Embed the windows of every complete day and store the coordinates in
aitana-storage, replacing those of the last fit.
"""

import pandas as pd
import xarray as xr
from aitana.anomaly.snet import embed
from aitana.storage import Storage

cfg = snakemake.config

frames = [xr.load_dataset(f) for f in snakemake.input]
frames = [f.sel(channel=cfg["channels"]) for f in frames if "order1" in f]
if not frames:
    raise ValueError("No three-component data in any day.")
snet = xr.concat(frames, dim="datetime")
# as in store.py: days without data become NaN rather than a hole in the time axis
snet = snet.reindex(
    datetime=pd.date_range(snet.datetime.values[0], snet.datetime.values[-1], freq="10min")
)
embedding = embed(snet)

net, sta, loc, cha = cfg["seed_id"].split(".")
store = Storage("ruapehu", cfg["storage_dir"]).get_substore(sta, loc, cha.replace("?", ""))
# (component, datetime): aitana-storage-api serves a 2-D feature's first dimension as freqs
store.save(embedding.transpose("component", "datetime").to_dataset(name="snet_umap"))
embedding.to_pandas().add_prefix("umap").to_csv(snakemake.output[0])
