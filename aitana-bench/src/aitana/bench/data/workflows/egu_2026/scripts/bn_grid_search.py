import numpy as np
import pandas as pd
from whakaaribn.grid_search import grid_search
from whakaaribn.model import WhakaariModel

model_class = WhakaariModel
data = pd.read_csv(snakemake.input[0], parse_dates=True, index_col=0)
params_gcv = [
    {
        "discretize__bins": [[0, 5, 100], [0, 50, 100], [0, 95, 100]],
        "clf__nstates": [2],
        "clf__pew": np.arange(10, 110, 10),
    },
    {
        "discretize__bins": [[0, 5, 95, 100], [0, 33, 66, 100], [0, 25, 75, 100]],
        "clf__nstates": [3],
        "clf__pew": np.arange(10, 110, 10),
    },
    {
        "discretize__bins": [
            [0, 25, 50, 75, 100],
            [0, 5, 50, 95, 100],
            [0, 10, 50, 90, 100],
            [0, 20, 50, 80, 100],
        ],
        "clf__nstates": [4],
        "clf__pew": np.arange(10, 110, 10),
    },
    {
        "discretize__bins": [
            [0, 20, 40, 60, 80, 100],
            [0, 5, 20, 80, 95, 100],
            [0, 5, 25, 75, 95, 100],
        ],
        "clf__nstates": [5],
        "clf__pew": np.arange(10, 110, 10),
    },
]
search_results = grid_search(
    data, params_gcv, recompute=True, njobs=10, model_class=model_class
)
search_results.to_csv(snakemake.output[0], index=False)
