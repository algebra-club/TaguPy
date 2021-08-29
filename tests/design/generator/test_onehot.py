"""
Test for One Hot Design Generator Module
"""

import numpy as np
import pytest

from tagupy.design.generator import OneHot


@pytest.fixture
def correct_inputs():
    return [1, 2, 3, 4, 5, 6, 7]


def test_init_invalid_input():
    arg = [
        ["moge", None, np.ones((2, 3)), 3.4, 0, -22]
    ]
    for el in arg:
        with pytest.raises(AssertionError) as e:
            OneHot(el)
        assert f"{el}" in f"{e.value}", \
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_init_correct_input(correct_inputs):
    exp = [1, 2, 3, 4, 5, 6, 7]
    for i, el in enumerate(correct_inputs):
        assert OneHot(el).n_rep == exp[i], \
            f"self.n_rep expected {exp[i]}, got {OneHot(el).n_rep}"


def test_get_exmatrix_invalid_input_dtype():
    arg = [
        ["moge", None, np.ones((2, 3)), 3.4, 0, -1]
    ]
    model = OneHot(1)
    for n_factor in arg:
        with pytest.raises(AssertionError) as e:
            model.get_exmatrix(n_factor)
        assert f"{n_factor}" in f"{e.value}", \
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_get_exmatrix_output_dtype(correct_inputs):
    model = OneHot(1)
    for n_factor in correct_inputs:
        ret = model.get_exmatrix(n_factor)
        assert isinstance(ret, np.ndarray), \
            f"dtype of exmatrix expected np.ndarray got {type(ret)}"


def test_get_exmatrix_output_shape(correct_inputs):
    n_rep = 11
    model = OneHot(n_rep)
    for n_factor in correct_inputs:
        exp = ((n_factor + 1) * n_rep, n_factor)
        ret = model.get_exmatrix(n_factor)
        assert ret.shape == exp, \
            f"shape of exmatrix expected {exp}, got {ret.shape}"


def test_get_exmatrix_output_element(correct_inputs):
    model = OneHot(11)
    for n_factor in correct_inputs:
        ret = model.get_exmatrix(n_factor)
        assert ((ret == 1) | (ret == 0)).all(), \
            f"all the elements in exmatrix should be either 0 or 1, got {ret}"


def test_get_exmatrix_sum(correct_inputs):
    n_rep = 11
    model = OneHot(n_rep)

    for n_factor in correct_inputs:
        ret = model.get_exmatrix(n_factor)
        sum = (ret.sum(axis=0), ret.sum(axis=1))

        assert ((sum[1] == 1) | (sum[1] == 0)).all(), \
            f"sum of values in a row should be \
                either 0 or 1, got {sum[1]}"
        assert np.bincount(sum[1])[0] == n_rep, \
            f"raws for negative control should be given as many as n_rep, \
                got {np.bincount(sum[1])[0]}"
        assert np.array_equal(sum[0], np.full((n_factor), n_rep)), \
            f"sum of values in a col should be n_rep, got {sum[0]}"
