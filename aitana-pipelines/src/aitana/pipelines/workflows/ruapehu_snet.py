from pathlib import Path

from aitana.pipelines.cli import WorkflowDescriptor
from aitana.pipelines import get_data

workflow = WorkflowDescriptor(
    name="ruapehu_snet",
    volcano="ruapehu",
    description="Scattering-network coefficients of three-component Ruapehu seismic data, their UMAP embedding and novelty index, stored for Grafana",
    workflowdir=Path(get_data("data/workflows/ruapehu_snet/")),
    outputs={
        "snet": "snet/",
        "model": "model/",
        "results": "results/",
        "storage": "storage/",
    },
)
