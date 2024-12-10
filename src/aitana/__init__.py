import importlib
import os
from typing import Optional
from .tilde import tilde_request


def get_data(filename: Optional[os.PathLike] = None) -> str:
    """Get data from the Aitana dataset.

    Parameters
    ----------
    filename : str, optional
        Name of the file to read. If None, the whole dataset is returned.

    Returns
    -------
    pd.DataFrame
        The requested data.

    Examples
    --------
    >>> from aitana import get_data
    >>> get_data()
    """
    f = importlib.resources.files(__package__)
    return str(f) if filename is None else str(os.path.join(f, filename))
