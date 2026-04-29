import shutil
import subprocess
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Protocol, Sequence

import typer

from aitana.logging_config import get_logger

logger = get_logger(__name__)

app = typer.Typer()


@dataclass
class WorkflowDescriptor:
    name: str
    volcano: str
    description: str
    workflowdir: Path
    outputs: dict[str, str]  # logical name → relative output path


class PipelineBackend(Protocol):
    def execute(
        self, target: str, cores: int, extra: Sequence[str] | None = None
    ) -> None: ...
    def clean(self) -> None: ...


class PipelineError(RuntimeError):
    pass


def discover_workflows() -> dict[str, WorkflowDescriptor]:
    result = {}
    for ep in entry_points(group="volcanobench.workflows"):
        try:
            result[ep.name] = ep.load()
        except Exception as e:
            logger.warning("Failed to load workflow '%s': %s", ep.name, e)
    return result


class SnakemakeBackend:
    def __init__(
        self, workflowdir: Path, outdir: Path, default_args: tuple[str, ...] = ()
    ):
        self.workflowdir = workflowdir
        self.outdir = outdir
        self.default_args = default_args
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.snakefile = self._prepare_workflow_directory()

    def _prepare_workflow_directory(self) -> str:
        """Copy the snakefile, rules/, and notebooks/ into *directory* if it differs
        from the bundled workflow directory.  Returns the path to the snakefile that
        should be passed to snakemake (the copy when applicable, original otherwise).
        """
        if self.outdir.resolve() == self.workflowdir.resolve():
            raise PipelineError(
                "Output directory cannot be the same as the workflow directory."
            )
        shutil.copytree(self.workflowdir, self.outdir, dirs_exist_ok=True)
        return str(self.outdir / "Snakefile")

    def _base_cmd(self) -> list[str]:
        return [
            "snakemake",
            "--snakefile",
            str(self.snakefile),
            "--directory",
            str(self.outdir),
        ]

    def _unlock(self) -> None:
        cmd = self._base_cmd() + ["--unlock"]
        try:
            subprocess.run(cmd, check=False, capture_output=True)
        except Exception as e:
            logger.warning("snakemake --unlock failed: %s", e)

    def execute(self, cores: int, extra: Sequence[str] | None = None) -> None:
        self._unlock()
        cmd = self._base_cmd() + ["--cores", str(cores)]
        cmd.extend(self.default_args)
        if extra:
            cmd.extend(extra)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise PipelineError("Snakemake failed.")

    def clean(self) -> None:
        cmd = self._base_cmd() + ["--cores", "1", "--delete-all-output"]
        cmd.extend(self.default_args)
        subprocess.run(cmd, check=True)


def _resolve(volcano: str) -> list[WorkflowDescriptor]:
    matches = [w for w in discover_workflows().values() if w.volcano == volcano]
    if not matches:
        typer.echo(f"No workflows registered for '{volcano}'", err=True)
        raise typer.Exit(1)
    return matches


@app.command()
def run(volcano: str, outdir: str, cores: int = 1):
    """Run all registered benchmark workflows for a volcano."""
    for w in _resolve(volcano):
        SnakemakeBackend(w.workflowdir, Path(outdir)).execute(cores=cores)


@app.command()
def clean(volcano: str, outdir: str):
    """Delete all workflow outputs for a volcano."""
    for w in _resolve(volcano):
        SnakemakeBackend(w.workflowfile, Path(outdir)).clean()


@app.command(name="list")
def list_workflows():
    """List all registered workflows."""
    workflows = discover_workflows()
    if not workflows:
        typer.echo("No workflows registered.")
        return
    for w in workflows.values():
        typer.echo(f"{w.name} ({w.volcano}): {w.description}")


if __name__ == "__main__":
    app()
