import importlib
from os import PathLike
from typing import Optional

from .dsar import DSAR
from .feature_base import FeatureBaseClass
from .rsam import RSAM
from .spectral_features import SpectralFeatures
from .ssam import SSAM


def get_data(filename: Optional[PathLike] = None) -> str:
    """Return path to aitana.features package.

    Parameters
    ----------
    filename : Pathlike, default None
        Append `filename` to returned path.

    Returns
    -------
    pkgdir_path

    """
    f = importlib.resources.files(__package__)
    return str(f) if filename is None else str(f / filename)
