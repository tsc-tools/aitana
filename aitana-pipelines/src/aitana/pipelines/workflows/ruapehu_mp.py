from pathlib import Path

from aitana.pipelines.cli import WorkflowDescriptor
from aitana.pipelines import get_data

workflow = WorkflowDescriptor(
    name="ruapehu_mp",
    volcano="ruapehu",
    description="Matrix-profile anomaly index and alarm series of Ruapehu RSAM, stored for Grafana",
    workflowdir=Path(get_data("data/workflows/ruapehu_mp/")),
    outputs={
        "rsam": "rsam/",
        "results": "results/",
        "storage": "storage/",
    },
)
