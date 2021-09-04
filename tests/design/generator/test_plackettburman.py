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


def test_get_exmatrix_invalid_input_dtype():
    arg = [
        ["moge", None, np.ones((2, 3)), 3.4, 0, -1]
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
            f"dtype of exmatrix expected np.ndarray got {type(ret)}"
