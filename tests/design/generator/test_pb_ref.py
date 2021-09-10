"""
Test for Plackett-Burman Refernce Module
"""

import numpy as np
import pytest

from tagupy.design.generator._pb_ref import _pb


@pytest.fixture
def supported_num():
    # multiples of 4 within [4, 100] but 92
    l_num = [i for i in np.arange(4, 90, 4)] + [96, 100]
    return [(i, _pb(i)) for i in l_num]


def test_pb92_not_implemented_error():
    with pytest.raises(ValueError):
        _pb(92)


def test_pbX_output(supported_num):
    for num_fn in supported_num:
        num = num_fn[0]
        matrix = num_fn[1]
        # dtype of output
        assert isinstance(matrix, np.ndarray), \
            f"dtype of retrun var expected numpy.ndarray, got {type(matrix)} in _pb{num}"
        # shape of output
        assert matrix.shape == (num, num - 1), \
            f"shape of retrun var expected (X, X-1) for _pbX, got {matrix.shape} in _pb{num}"


def test_pbX_conversion_to_hadamard(supported_num):
    for num_fn in supported_num:
        num = num_fn[0]
        pb = num_fn[1]
        hadamard = np.concatenate([pb, np.ones((num, 1))], axis=1)
        assert ((pb == 1) | (pb == -1)).all(), \
            f"all the elements in exmatrix should be either -1 or 1, got {pb} in _pb{num}"
        assert (hadamard @ hadamard.T == num * np.identity(num)).all(), \
            f"return var expected to be submatrix of hadamard matrix, got {pb} in _pb{num}"
