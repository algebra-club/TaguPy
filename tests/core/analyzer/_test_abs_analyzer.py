"""Test of _Analyzer, super class of every statistic analysis module"""
import numpy as np
import pytest

from tagupy.core import _Analyzer


def test_constructor():
    arg = {
        "exmatrix": np.zeros((5, 5)),
        "result": np.ones((5, 5))
    }
    temp = _Analyzer(**arg)
    assert hasattr(temp, 'exmatrix'), 'exmatrix is not defined'
    assert hasattr(temp, 'result'), 'result is not defined'


def test_constructor_assertion():
    arg = {
        "exmatrix": np.zeros((5, 5)),
        "result": np.ones((5, 6))
    }

    with pytest.raises(AssertionError):
        _Analyzer(**arg)
