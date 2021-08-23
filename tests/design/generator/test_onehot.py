"""
Test for One Hot Design Generator Module
"""

import numpy as np
import pytest

from tagupy.core.generator import OneHot


def test_init_input_dtype():
    arg = ["moge", None, np.ones((2, 3))]
    with pytest.raises(ValueError):
        [OneHot(i) for i in arg]


def test_init_input_value():
    arg = [0, -1, -3.5]
    with pytest.raises(AssertionError):
        [OneHot(i) for i in arg]


def test_init_correct_input():
    arg = [1, 2, 3, 4, 5, 6, 7.8]
    exp = [1, 2, 3, 4, 5, 6, 7]
    _model = [OneHot(i) for i in arg]
    l_rep = [i.n_rep for i in _model]
    assert l_rep == exp, \
        "Error: something is wrong with n_rep"


def test_exmatrix_input_dtype():
    arg = ["moge", None, np.ones((2, 3))]
    _model = OneHot(1)
    with pytest.raises(ValueError):
        [_model.get_exmatrix(i) for i in arg]


def test_exmatrix_input_value():
    arg = [1, 0, -1, -3.5]
    _model = OneHot(1)
    with pytest.raises(AssertionError):
        [_model.get_exmatrix(i) for i in arg]


def test_exmatrix_dtype__n_f():
    arg = [2, 3, 4, 5, 6]
    _model = OneHot(1)
    _res = [isinstance(
        _model.get_exmatrix(i),
        np.ndarray) for i in arg]
    assert all(_res), \
        "Error: dtype of exmatrix is not np.ndarray"


def test_exmatrix_dtype__n_r():
    arg = [OneHot(i) for i in (1, 2, 3, 4, 5)]
    _res = [isinstance(
        i.get_exmatrix(2),
        np.ndarray) for i in arg]
    assert all(_res), \
        "Error: dtype of exmatrix is not np.ndarray"


def test_exmatrix_shape__n_f():
    arg = [2, 3, 4, 5, 6]
    _model = OneHot(11)
    _temp = [_model.get_exmatrix(i) for i in arg]
    _res = [v.shape == (
        arg[i] * 11,
        arg[i]
        ) for i, v in enumerate(_temp)]
    assert all(_res), \
        "Error: something is wrong with exmatrix shape"


def test_exmatrix_shape__n_r():
    l_rep = [1, 2, 3, 4, 5]
    arg = [OneHot(i) for i in l_rep]
    _temp = [i.get_exmatrix(11) for i in arg]
    _res = [v.shape == (
        11 * l_rep[i],
        11
        ) for i, v in enumerate(_temp)]
    assert all(_res), \
        "Error: something is wrong with exmatrix shape"


def test_exmatrix_sum_row():
    arg = [2, 3, 4, 5, 6]
    _model = OneHot(1)
    _temp = [_model.get_exmatrix(i) for i in arg]
    _res = [(
        np.sum(v, axis=1) == np.ones(arg[i])
        ).all() for i, v in enumerate(_temp)]
    assert all(_res), \
        "Error: sum of values in a row shoud be 1"


def test_exmatrix_sum_col():
    l_rep = [1, 2, 3, 4, 5]
    arg = [OneHot(i) for i in l_rep]
    _temp = [i.get_exmatrix(11) for i in arg]
    _res = [(
        np.sum(v, axis=0) == l_rep[i] * np.ones(11)
        ).all() for i, v in enumerate(_temp)]
    assert all(_res), \
        "Error: sum of values in a col should be n_rep"


def test_alias_dtype__n_f():
    arg = [2, 3, 4, 5, 6]
    _model = [OneHot(1) for i in arg]
    [v.get_exmatrix(arg[i]) for i, v in enumerate(_model)]
    _res = [isinstance(
        i.get_alias_matrix(),
        np.ndarray) for i in _model]
    assert all(_res), \
        "Error: dtype of alias matrix is not np.ndarray"


def test_alias_dtype__n_r():
    arg = [OneHot(i) for i in (1, 2, 3, 4, 5)]
    [i.get_exmatrix(11) for i in arg]
    _res = [isinstance(
        i.get_alias_matrix(),
        np.ndarray) for i in arg]
    assert all(_res), \
        "Error: dtype of alias matrix is not np.ndarray"

# alias matrixの実装について相談したいので、一旦保留にします
# def test_alias_matrix():
#     n_rep = 1
#     n_factor = 3
#     _model = OneHot(n_rep)
#     _model.get_exmatrix(n_factor)
#     exp = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
#     _res = _model.get_alias_matrix()
#     assert (exp == _res).all(), \
#         "Error: something is wrong with alias_matrix"
