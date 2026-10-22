"""
Matrix-profile anomaly index of log10 RSAM, the raw matrix profile it ranks, its
alarm series (1 where the index reaches the threshold) and the threshold, stored
for Grafana. Each window (the last lookback_hours, plus one per day while building
the archive) is scored against its own trailing history.
"""

import warnings

import numpy as np
import pandas as pd
from aitana.anomaly.matrix_profile import MatrixProfile
from aitana.storage import Storage

MAX_GAP = 6  # interpolate gaps of up to one hour

cfg = snakemake.config
windows = [[pd.Timestamp(t) for t in window] for window in snakemake.params.windows]

frames = [pd.read_csv(f, index_col=0, parse_dates=True) for f in snakemake.input]
rsam = pd.concat([f for f in frames if len(f) > 0])["rsam"]
rsam = rsam[~rsam.index.duplicated(keep="last")]

detector = MatrixProfile(window=cfg["window"])


def score(history_start, score_start, end):
    """
    Score one window of log10 RSAM against the history before it.

    Parameters
    ----------
    history_start : pd.Timestamp
        First sample of the history.
    score_start : pd.Timestamp
        First scored sample.
    end : pd.Timestamp
        End of the window (exclusive).

    Returns
    -------
    pd.DataFrame
        ``anomaly_index`` and ``matrix_profile`` from ``score_start`` to ``end``.
    """
    x = rsam.reindex(pd.date_range(history_start, end, freq="10min", inclusive="left"))
    x = np.log10(x.where(x > 0))
    gap = x.isna()
    gap_length = gap.groupby((~gap).cumsum()).transform("sum")
    x = x.interpolate(limit_area="inside").where(~gap | (gap_length <= MAX_GAP))
    index = detector.compute_anomaly_index(x, score_start)
    return pd.DataFrame(
        {"anomaly_index": index, "matrix_profile": detector.profile_}
    ).loc[score_start:]


scored = []
for history_start, score_start, end in windows:
    try:
        scored.append(score(history_start, score_start, end))
    except ValueError as error:  # no usable history, e.g. after a long station outage
        warnings.warn(f"Not scored from {score_start} to {end}: {error}")
if not scored:
    raise ValueError("No window had any usable history.")
scored = pd.concat(scored)
scored = scored[~scored.index.duplicated(keep="last")].sort_index()

rsam = rsam.reindex(
    pd.date_range(
        min(w[0] for w in windows), max(w[2] for w in windows), freq="10min", inclusive="left"
    )
)
index = scored["anomaly_index"]
profile = scored["matrix_profile"]
alarm = (index >= cfg["threshold"]).astype(float).where(index.notna())
# the threshold in force when each sample was scored, so retunes stay traceable
threshold = pd.Series(float(cfg["threshold"]), index=index.index)

pd.DataFrame(
    {"rsam": rsam, "anomaly_index": index, "matrix_profile": profile, "alarm": alarm}
).rename_axis(
    "datetime"
).to_csv(snakemake.output.index)

net, sta, loc, cha = cfg["seed_id"].split(".")
store = Storage("ruapehu", cfg["storage_dir"]).get_substore(sta, loc, cha)
store.save(rsam.rename_axis("datetime").to_xarray().to_dataset(name="rsam"))
store.save(index.rename_axis("datetime").to_xarray().to_dataset(name="rsam_mp_index"))
store.save(profile.rename_axis("datetime").to_xarray().to_dataset(name="rsam_mp_profile"))
store.save(alarm.rename_axis("datetime").to_xarray().to_dataset(name="rsam_mp_alarm"))
store.save(threshold.rename_axis("datetime").to_xarray().to_dataset(name="rsam_mp_threshold"))
