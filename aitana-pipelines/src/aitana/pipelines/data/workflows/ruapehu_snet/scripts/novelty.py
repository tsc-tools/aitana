"""
Novelty index of this run's windows against the reference days, the distance it
ranks, its alarm series (1 where the index reaches the threshold) and the
threshold, stored for Grafana.
"""

import joblib
import pandas as pd
import xarray as xr
from aitana.storage import Storage

cfg = snakemake.config

frames = [xr.load_dataset(f) for f in snakemake.input.snet]
frames = [f for f in frames if "order1" in f]
if not frames:
    raise ValueError("No three-component data in any day of this run.")
snet = xr.concat(frames, dim="datetime")
# as in store.py: days without data become NaN rather than a hole in the time axis
snet = snet.reindex(
    datetime=pd.date_range(snet.datetime.values[0], snet.datetime.values[-1], freq="10min")
)
detector = joblib.load(snakemake.input.model)
index = detector.compute_anomaly_index(snet)
distance = detector.distance_
alarm = (index >= cfg["threshold"]).astype(float).where(index.notna())
# the threshold in force when each window was scored, so retunes stay traceable
threshold = pd.Series(float(cfg["threshold"]), index=index.index)

pd.DataFrame(
    {"anomaly_index": index, "distance": distance, "alarm": alarm}
).rename_axis("datetime").to_csv(snakemake.output[0])

net, sta, loc, cha = cfg["seed_id"].split(".")
store = Storage("ruapehu", cfg["storage_dir"]).get_substore(sta, loc, cha.replace("?", ""))
for name, series in [("index", index), ("distance", distance), ("alarm", alarm),
                     ("threshold", threshold)]:
    store.save(series.rename_axis("datetime").to_xarray().to_dataset(name=f"snet_novelty_{name}"))
