from pathlib import Path

from aitana.pipelines.cli import WorkflowDescriptor
from aitana.pipelines import get_data

workflow = WorkflowDescriptor(
    name="ruapehu_snet",
    volcano="ruapehu",
    description="Scattering-network coefficients of three-component Ruapehu seismic data, stored for detectors",
    workflowdir=Path(get_data("data/workflows/ruapehu_snet/")),
    outputs={
        "snet": "snet/",
        "results": "results/",
        "storage": "storage/",
    },
)
