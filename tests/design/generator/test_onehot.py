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
    _model = OneHot(1)
    for i in correct_inputs:
        assert isinstance(
            _model.get_exmatrix(i),
            np.ndarray
            ), \
                f"dtype of exmatrix expected np.ndarray \
                    got {type(_model.get_exmatrix(i))}"


def test_get_exmatrix_output_shape(correct_inputs):
    _model = OneHot(11)
    for v in correct_inputs:
        assert _model.get_exmatrix(v).shape == (
            (v + 1) * 11,
            v
            ), \
                f"shape of exmatrix expected \
                    ((n_factor + 1) * n_rep, n_facttor), \
                    got {_model.get_exmatrix(v).shape}"


def test_get_exmatrix_output_element(correct_inputs):
    _model = OneHot(11)
    for v in correct_inputs:
        assert np.logical_or(
            _model.get_exmatrix(v) == 1,
            _model.get_exmatrix(v) == 0
            ).all(), \
                f"all the elements in exmatrix should \
                    be either 0 or 1, got {_model.get_exmatrix(v)}"


def test_get_exmatrix_sum(correct_inputs):
    _model = OneHot(11)
    for v in correct_inputs:
        assert np.logical_or(
            np.sum(_model.get_exmatrix(v), axis=1) == 1,
            np.sum(_model.get_exmatrix(v), axis=1) == 0
        ).all(), \
            f"sum of values in a row should be \
                either 0 or 1, got {_model.get_exmatrix(v)}"
        assert np.sum(_model.get_exmatrix(v)) == v * 11, \
            f"raws for negative control should be given as many as n_rep, \
                got {_model.get_exmatrix(v)}"
        np.testing.assert_array_equal(
            np.sum(_model.get_exmatrix(v), axis=0),
            np.full((v), 11),
            f"sum of values in a col should be n_rep, got {_model.get_exmatrix(v)}"
        )
