import pytest

from aitana.features import FeatureBaseClass

def test_exceptions():
    """
    Test that the right exceptions are raised.
    """
    bc = FeatureBaseClass()
    with pytest.raises(NotImplementedError):
        bc.compute()
