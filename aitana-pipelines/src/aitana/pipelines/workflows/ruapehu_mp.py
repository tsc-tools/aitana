from pathlib import Path

from aitana.pipelines.cli import WorkflowDescriptor
from aitana.pipelines import get_data

workflow = WorkflowDescriptor(
    name="ruapehu_mp",
    volcano="ruapehu",
    description="Matrix-profile anomaly index of Ruapehu RSAM with an alert summary for aitana-bot",
    workflowdir=Path(get_data("data/workflows/ruapehu_mp/")),
    outputs={
        "rsam": "rsam/",
        "results": "results/",
        "storage": "storage/",
    },
)
