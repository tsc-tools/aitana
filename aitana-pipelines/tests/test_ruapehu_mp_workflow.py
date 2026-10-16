"""Offline checks of the ruapehu_mp workflow's job graph (Snakemake dry runs)."""

import re

import pytest

from aitana.pipelines.cli import SnakemakeBackend
from aitana.pipelines.workflows.ruapehu_mp import workflow


def planned_jobs(output):
    """
    Parse the job counts from a Snakemake dry-run "Job stats" table.

    Parameters
    ----------
    output : str
        Captured stdout/stderr of the dry run.

    Returns
    -------
    dict of str to int
        Number of planned jobs per rule.
    """
    return {rule: int(n) for rule, n in re.findall(r"^(\w+)\s+(\d+)$", output, re.M)}


@pytest.mark.parametrize(
    "end, partial",
    [("2022-03-14T00:00", False), ("2022-03-14T05:30", True)],
)
def test_job_graph(tmp_path, capfd, end, partial):
    SnakemakeBackend(workflow.workflowdir, tmp_path).execute(
        cores=1, extra=["-n", "--config", f"end={end}"]
    )
    jobs = planned_jobs(capfd.readouterr().out)
    # 28 days of history plus the 6-h window lead and the scored day
    assert jobs["rsam_day"] == 30
    assert jobs.get("rsam_partial", 0) == int(partial)
    assert jobs["matrix_profile"] == 1
    assert jobs["plot"] == 1
