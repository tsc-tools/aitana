"""
Matrix-profile anomaly index of log10 RSAM and the alert summary for aitana-bot.
"""

import json

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
scored = index[score_start:]
peak_time = scored.idxmax() if scored.notna().any() else None

pd.DataFrame({"rsam": rsam, "anomaly_index": index}).rename_axis("datetime").to_csv(
    snakemake.output.index
)

alert = {
    "volcano": "Ruapehu",
    "seed_id": cfg["seed_id"],
    "end": end.isoformat(),
    "score_start": score_start.isoformat(),
    "history_start": history_start.isoformat(),
    "window_hours": cfg["window"] * 10 / 60,
    "threshold": cfg["threshold"],
    "peak_index": None if peak_time is None else float(scored.max()),
    "peak_time": None if peak_time is None else peak_time.isoformat(),
    "alarm": peak_time is not None and bool(scored.max() >= cfg["threshold"]),
    "plot": "rsam_mp.png",
}
with open(snakemake.output.alert, "w") as fh:
    json.dump(alert, fh, indent=2)

net, sta, loc, cha = cfg["seed_id"].split(".")
store = Storage("ruapehu", cfg["storage_dir"]).get_substore(sta, loc, cha)
store.save(rsam.rename_axis("datetime").to_xarray().to_dataset(name="rsam"))
store.save(scored.rename_axis("datetime").to_xarray().to_dataset(name="rsam_mp_index"))
