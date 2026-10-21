"""
Matrix-profile anomaly index of log10 RSAM, its alarm series (1 where the index
reaches the threshold) and the threshold, stored for Grafana.
"""

import numpy as np
import pandas as pd
from aitana.anomaly.matrix_profile import MatrixProfile
from aitana.storage import Storage

MAX_GAP = 6  # interpolate gaps of up to one hour

cfg = snakemake.config
end = pd.Timestamp(snakemake.params.end)
score_start = pd.Timestamp(snakemake.params.score_start)
history_start = pd.Timestamp(snakemake.params.history_start)

frames = [pd.read_csv(f, index_col=0, parse_dates=True) for f in snakemake.input]
rsam = pd.concat([f for f in frames if len(f) > 0])["rsam"]
rsam = rsam[~rsam.index.duplicated(keep="last")]
rsam = rsam.reindex(pd.date_range(history_start, end, freq="10min", inclusive="left"))

x = np.log10(rsam.where(rsam > 0))
gap = x.isna()
gap_length = gap.groupby((~gap).cumsum()).transform("sum")
x = x.interpolate(limit_area="inside").where(~gap | (gap_length <= MAX_GAP))

index = MatrixProfile(window=cfg["window"]).compute_anomaly_index(x, score_start)
alarm = (index >= cfg["threshold"]).astype(float).where(index.notna())
scored = index[score_start:]
scored_alarm = alarm[score_start:]
# the threshold in force when each sample was scored, so retunes stay traceable
threshold = pd.Series(float(cfg["threshold"]), index=scored.index)

pd.DataFrame({"rsam": rsam, "anomaly_index": index, "alarm": alarm}).rename_axis(
    "datetime"
).to_csv(snakemake.output.index)

net, sta, loc, cha = cfg["seed_id"].split(".")
store = Storage("ruapehu", cfg["storage_dir"]).get_substore(sta, loc, cha)
store.save(rsam.rename_axis("datetime").to_xarray().to_dataset(name="rsam"))
store.save(scored.rename_axis("datetime").to_xarray().to_dataset(name="rsam_mp_index"))
store.save(scored_alarm.rename_axis("datetime").to_xarray().to_dataset(name="rsam_mp_alarm"))
store.save(threshold.rename_axis("datetime").to_xarray().to_dataset(name="rsam_mp_threshold"))
