"""Run the examples in the aitana.features docstrings."""

import doctest
import importlib
import pkgutil

import pytest

import aitana.features

MODULES = sorted(m.name for m in pkgutil.iter_modules(aitana.features.__path__))


@pytest.mark.parametrize("name", MODULES)
def test_docstrings(name):
    module = importlib.import_module(f"aitana.features.{name}")
    result = doctest.testmod(module)
    if result.attempted == 0:
        pytest.skip("no examples")
    assert result.failed == 0
