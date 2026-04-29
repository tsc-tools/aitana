from datetime import datetime
from functools import partial

import matplotlib.pyplot as plt
from tonik import Storage

from aitana.scoring import evaluate_threshold
from aitana.visualise import scoring_plot, set_style

set_style()

st = Storage("whakaari_forecasts", rootdir="./results")
st.starttime = datetime(2013, 1, 1)
st.endtime = datetime(2020, 1, 1)
bn_forecasts = st("bayesian_network").to_pandas()

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
scoring_function = partial(evaluate_threshold, pew=None, return_windows=True)
for ax, threshold in zip(axes, [0.3, 0.6]):
    _, legend_handles = scoring_plot(bn_forecasts, threshold, scoring_function, ax=ax)
    ax.set_title(f"Threshold = {threshold}")

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(legend_handles),
    bbox_to_anchor=(0.5, -0.04),
    frameon=False,
)
fig.savefig(snakemake.output[0], bbox_inches="tight", dpi=300)
