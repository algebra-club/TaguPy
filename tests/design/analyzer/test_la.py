'''
Test for Linear Analysis Module
'''

import numpy as np
import pytest
import statsmodels.api as sm
from tagupy.design.analyzer import LinearAnalysis
from typing import Callable, NamedTuple


@pytest.fixture
def correct_input():
    return []


def test_init_invalid_input():
    arg = [0, 4.5, np.array([1, 1, 1]), None, "hoge"]
    for v in arg:
        with pytest.raises(AssertionError) as e:
            LinearAnalysis(v)
        assert f"{v}" in f"{e.value}", \
        f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_init_correct_input():
    arg = ["OLS", "WLS", "GLS"]
    exp = [sm.OLS, sm.WLS, sm.GLS]
    for i, v in enumerate(arg):
        model = LinearAnalysis(v)
        assert exp[i] in model.model, \
            f"self.model expected to contain {v}, got {model.model}"
