from pathlib import Path

from aitana import WorkflowDescriptor, get_data

workflow = WorkflowDescriptor(
    name="egu2026",
    volcano="whakaari",
    description="Bayesian network vs decision tree benchmark for Whakaari/White Island (EGU 2026)",
    workflowdir=Path(get_data("data/workflows/egu_2026/")),
    outputs={
        "download": "data/whakaari_data.csv",
        "benchmark": "results/whakaari_forecasts/",
        "plot": "plots/forecast_comparison.png",
    },
)
