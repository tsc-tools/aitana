from datetime import datetime

import pandas as pd
from tonik import Storage

from aitana.visualise import set_style, trellis_plot

set_style()

st = Storage("whakaari_forecasts", rootdir="results")
st.starttime = datetime(2012, 1, 1)
st.endtime = datetime(2020, 1, 1)
models = {
    "Bayesian Network": {"model": st("bayesian_network")},
    "Random Forest": {"model": st("random_forest").interpolate_na(dim="datetime")},
}
data = pd.read_csv(snakemake.input[2], parse_dates=True, index_col=0)
fig = trellis_plot(models, data=data, groups=["b", "c", "d"])
fig.savefig(snakemake.output[0], bbox_inches="tight", dpi=300)
