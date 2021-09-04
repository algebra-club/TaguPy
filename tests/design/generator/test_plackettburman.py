"""
Test for Plackett-Burman Design Generator Module
"""

import numpy as np
import pytest

from tagupy.design.generator import PlackettBurman


@pytest.fixture
def correct_input():
    return [1, 2, 5, 10, 40, 80, 97]


def test_init_invalid_input():
    arg = [
        ["moge", None, np.ones((2, 3)), 3.4, 0, -22]
    ]
    for el in arg:
        with pytest.raises(AssertionError) as e:
            PlackettBurman(el)
        assert f"{el}" in f"{e.value}", \
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_init_correct_input(correct_input):
    exp = [1, 2, 5, 10, 40, 80, 97]
    for i, el in enumerate(correct_input):
        assert PlackettBurman(el).n_rep == exp[i], \
            f"self.n_rep expected {exp[i]}, got {PlackettBurman(el).n_rep}"


def test_get_exmatrix_invalid_input():
    arg = [
        ["moge", None, np.ones((2, 3)), 3.4, 0, -1, 100, 88, 89, 90, 91]
    ]
    model = PlackettBurman(1)
    for n_factor in arg:
        with pytest.raises(AssertionError) as e:
            model.get_exmatrix(n_factor)
        assert f"{n_factor}" in f"{e.value}", \
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_get_exmatrix_output_dtype(correct_input):
    model = PlackettBurman(1)
    for n_factor in correct_input:
        ret = model.get_exmatrix(n_factor)
        assert isinstance(ret, np.ndarray), \
            f"dtype of exmatrix expected numpy.ndarray got {type(ret)}"


def test_get_matrix_output_shape(correct_input):
    n_rep = 11
    model = PlackettBurman(n_rep)
    for n_factor in correct_input:
        n_run = 4 * (n_factor // 4 + 1)
        exp = (n_run * n_rep, n_factor)
        ret = model.get_exmatrix(n_factor)
        assert ret.shape == exp, \
            f"shape of exmatrix expected {exp}, got {ret.shape}"


def test_get_exmatrix_output_element(correct_input):
    model = PlackettBurman(11)
    for n_factor in correct_input:
        ret = model.get_exmatrix(n_factor)
        assert ((ret == 1) | (ret == -1)).all(), \
            f"all the elements in exmatrix should be either -1 or 1, got {ret}"
