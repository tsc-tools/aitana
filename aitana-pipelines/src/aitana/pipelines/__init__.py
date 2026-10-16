import importlib
import os
from typing import Optional


def get_data(filename: Optional[os.PathLike] = None) -> str:
    """Get path to bundled data in the aitana.pipelines package (workflow files, etc.)."""
    f = importlib.resources.files(__package__)
    return str(f) if filename is None else str(os.path.join(f, filename))
