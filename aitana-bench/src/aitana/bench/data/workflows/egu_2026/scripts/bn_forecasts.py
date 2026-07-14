import pandas as pd
from whakaaribn.forecast import forecast
from whakaaribn.grid_search import get_best_estimator

data = pd.read_csv(snakemake.input[0], parse_dates=True, index_col=0)
bins, pew = get_best_estimator(snakemake.input[1])
xds = forecast(data, bins=bins, pew=pew, smoothing=30, exclude_from_test=[])
xds.to_netcdf(snakemake.output[0])
