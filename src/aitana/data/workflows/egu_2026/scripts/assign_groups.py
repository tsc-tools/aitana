import pandas as pd
from whakaaribn.util import assign_group_labels

from aitana import whakaari

data = pd.read_csv(snakemake.input[0], parse_dates=True, index_col=0)
eruptions = whakaari.eruptions(2, "0D", end_date=data.index[-1])
data_with_groups = assign_group_labels(
    data,
    eruptions,
    startdate=data.index[0],
    enddate=data.index[-1],
    ndays=30,
    min_interval=360,
)
data_with_groups.to_csv(snakemake.output[0])
