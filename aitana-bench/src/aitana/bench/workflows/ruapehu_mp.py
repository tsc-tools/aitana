from pathlib import Path

from aitana.bench.volcanobench import WorkflowDescriptor
from aitana.bench import get_bench_data

workflow = WorkflowDescriptor(
    name="ruapehu_mp",
    volcano="ruapehu",
    description="Matrix-profile anomaly index of Ruapehu RSAM with an alert summary for aitana-bot",
    workflowdir=Path(get_bench_data("data/workflows/ruapehu_mp/")),
    outputs={
        "rsam": "rsam/",
        "results": "results/",
        "storage": "storage/",
    },
)
