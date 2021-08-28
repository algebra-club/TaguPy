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
        ["moge", None, np.ones((2, 3)), 3.4],
        [0, -22]
    ]
    for el in arg[0]:
        with pytest.raises(AssertionError) as e:
            OneHot(el)
        assert f"{type(el)}" in f"{e.value}", \
            f"n_rep expected int, got {type(el)}"
    for el in arg[1]:
        with pytest.raises(AssertionError) as e:
            OneHot(el)
        assert f"n_rep expected >= 1, got {el}"


def test_init_correct_input(correct_inputs):
    exp = [1, 2, 3, 4, 5, 6, 7]
    for i, v in enumerate(correct_inputs):
        assert OneHot(v).n_rep == exp[i], \
            f"self.n_rep expected {exp[i]}, \
                got {OneHot(v).n_rep}"


def test_get_exmatrix_invalid_input_dtype():
    arg = [
        ["moge", None, np.ones((2, 3)), 3.4],
        [0, -1]
    ]
    _model = OneHot(1)
    for el in arg[0]:
        with pytest.raises(AssertionError) as e:
            _model.get_exmatrix(el)
        assert f"{type(el)}" in f"{e.value}", \
            f"n_rep expected int, got {type(el)}"
    for el in arg[1]:
        with pytest.raises(AssertionError) as e:
            _model.get_exmatrix(el)
        assert f"n_rep expected >= 1, got {el}"


def test_get_exmatrix_output_dtype(correct_inputs):
    model = OneHot(1)
    for v in correct_inputs:
        ret = model.get_exmatrix(v)
        assert isinstance(ret, np.ndarray), \
            f"dtype of exmatrix expected np.ndarray got {type(ret)}"


def test_get_exmatrix_output_shape(correct_inputs):
    model = OneHot(11)
    for v in correct_inputs:
        exp = ((v + 1) * 11, v)
        ret = model.get_exmatrix(v)
        assert ret.shape == exp, \
            f"shape of exmatrix expected {exp}, got {ret.shape}"


def test_get_exmatrix_output_element(correct_inputs):
    model = OneHot(11)
    for v in correct_inputs:
        ret = model.get_exmatrix(v)
        assert ((ret == 1) | (ret == 0)).all(), \
            f"all the elements in exmatrix should be either 0 or 1, got {ret}"


def test_get_exmatrix_sum(correct_inputs):
    n_rep = 11
    model = OneHot(n_rep)

    for n_factor in correct_inputs:
        ret = model.get_exmatrix(n_factor)
        sum = (np.sum(ret, axis=0), np.sum(ret, axis=1), np.sum(ret))

        assert ((sum[1] == 1) | (sum[1] == 0)).all(), \
            f"sum of values in a row should be \
                either 0 or 1, got {sum[1]}"
        assert sum[2] == n_factor * n_rep, \
            f"raws for negative control should be given as many as n_rep, \
                got {ret}"
        assert np.array_equal(sum[0], np.full((n_factor), n_rep)), \
            f"sum of values in a col should be n_rep, got {sum[0]}"
