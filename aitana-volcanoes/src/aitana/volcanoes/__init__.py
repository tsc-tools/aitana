import importlib
import os
from typing import Optional

from aitana.access.logging_config import setup_logging

setup_logging()


def get_data(filename: Optional[os.PathLike] = None) -> str:
    """Get path to bundled data in the aitana.volcanoes package.

    Parameters
    ----------
    filename : str, optional
        Relative path within the data directory. If None, returns the data directory root.

    Returns
    -------
    str
        Absolute path to the requested file or directory.
    """
    f = importlib.resources.files(__package__)
    return str(f) if filename is None else str(os.path.join(f, filename))
