"""
Test for Definitive Screening Design  module
"""

import numpy as np
import pytest

from tagupy.design.generator import DSD


@pytest.fixture
def correct_input():
    arg = [i for i in range(1, 49)]
    return arg


def test_init_valid_input(correct_input):
    for a, e in zip(correct_input, correct_input):
        init = DSD(a).n_rep
        assert init == e,\
            f'self.n_rep expected {e}, got {init}'


def test_init_invalid_input():
    arg = [
        ["moge", None, np.ones((2, 3)), 3.4, 0, -22]
    ]
    for a in arg:
        with pytest.raises(AssertionError) as e:
            DSD(a)
        assert f'{a}' in f"{e.value}",\
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_get_exmatrix_invalid_input():
    arg0 = ['moge', None, [], 0, -1, 1, 51, 3, 26]
    arg1 = ['moge', None, [], 0, -1, 1, 3, 51, 26]
    cor = [2 for i in range(9)]
    model = DSD(2)
    for a, c in zip(arg0[:5], cor[:5]):
        with pytest.raises(AssertionError) as e:
            model.get_exmatrix(n_fac=a, n_fake=c)
        assert f'{a}' in f'{e.value}',\
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"
    for c, a in zip(cor[:5], arg1[:5]):
        with pytest.raises(AssertionError) as e:
            model.get_exmatrix(n_fac=c, n_fake=a)
        assert f'{a}' in f'{e.value}',\
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"
    for a, c in zip(arg0[5:], arg1[5:]):
        with pytest.raises(AssertionError) as e:
            model.get_exmatrix(n_fac=a, n_fake=c)
        assert f'{a+c}' in f'{e.value}',\
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_get_exmatrix_valid_output(correct_input):
    n_fake = 2
    model = DSD(2)
    for fac in correct_input:
        sum_fac = fac + n_fake
        if fac % 2 == 1:
            sum_fac += 1
        ex_mat = model.get_exmatrix(n_fac=fac, n_fake=n_fake)
        assert isinstance(ex_mat, np.ndarray), \
            f'dtype of exmatrix expected numpy.ndarray got {type(ex_mat)}'
        cor = ((2 * sum_fac + 1) * 2, fac)
        ret = ex_mat.shape
        assert ret == cor, \
            f"shape of exmatrix expected {cor}, got {ret}"
        assert ((ex_mat == 0) | (ex_mat == 1) | (ex_mat == -1)).all(),\
            f'Error: all the elements in exmatrix should be either 0, -1, or 1, got {ex_mat}'
