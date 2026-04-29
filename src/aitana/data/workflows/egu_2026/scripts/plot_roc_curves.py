from datetime import datetime, timezone
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tonik import Storage

from aitana import whakaari
from aitana.scoring import evaluate_threshold, get_roc_curve, make_strictly_increasing
from aitana.visualise import set_style

set_style()

st = Storage("whakaari_forecasts", rootdir="results")
st.starttime = datetime(2012, 1, 1)
st.endtime = datetime(2020, 1, 1)
bn_forecasts = st("bayesian_network").to_pandas().to_frame(name="prob")
bn_forecasts.index = pd.to_datetime(bn_forecasts.index)
dt_forecasts = st("random_forest").to_pandas().to_frame(name="prob")
dt_forecasts.index = pd.to_datetime(dt_forecasts.index)

data_file = snakemake.input[0]
data = pd.read_csv(data_file, parse_dates=True, index_col=0)

rocs = {}
for name, fcst in zip(
    ["Bayesian Network", "Random Forest"], [bn_forecasts, dt_forecasts]
):
    fcst.index = fcst.index.tz_localize("utc")
    explosive_eruptions = whakaari.eruptions(2, "0D", end_date=fcst.index[-1]).loc[
        fcst.index[0] : fcst.index[-1]
    ]
    fcst["eruptions"] = explosive_eruptions.reindex(fcst.index, fill_value=0)[
        "Activity_Scale"
    ]
    thresholds = np.linspace(0.01, 0.99, 100)[::-1]
    tpr_bn, fpr_bn, precision_bn = get_roc_curve(
        fcst, thresholds, partial(evaluate_threshold, pew=None)
    )
    rocs[name] = dict(tpr=tpr_bn, fpr=fpr_bn, precision=precision_bn)

datatypes = ["RSAM", "CO2", "SO2", "H2S"]
for _ds in datatypes:
    _data = data[_ds].loc[
        st.starttime.replace(tzinfo=timezone.utc) : st.endtime.replace(
            tzinfo=timezone.utc
        )
    ]
    _data = _data.ffill()
    _data = pd.DataFrame({"prob": _data})
    _data.index = _data.index
    explosive_eruptions = whakaari.eruptions(2, "0D", end_date=_data.index[-1]).loc[
        _data.index[0] : _data.index[-1]
    ]
    _data["eruptions"] = explosive_eruptions.reindex(_data.index, fill_value=0)[
        "Activity_Scale"
    ]
    tpr_, fpr_, prec_ = get_roc_curve(
        _data, thresholds, partial(evaluate_threshold, pew=None)
    )
    rocs[_ds] = dict(tpr=tpr_, fpr=fpr_, precision=prec_)

# add roc curvses for all results in rocs

fig, ax = plt.subplots(figsize=(8, 6))
for label, roc in rocs.items():
    ax.plot(make_strictly_increasing(roc["fpr"]), roc["tpr"], label=label)
ax.plot([0, 1], [0, 1], "k--", label="Random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves for Whakaari Forecasts and Data")
ax.legend()
fig.savefig(snakemake.output[0], bbox_inches="tight", dpi=300)
