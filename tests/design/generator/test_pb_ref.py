"""
Test for Plackett-Burman Refernce Module
"""

import numpy as np
import pytest

import tagupy.design.generator._pb_ref as ref


@pytest.fixture
def supported_num():
    # multiples of 4 within [4, 100] but 92
    return [
        (4, ref._pb4), (8, ref._pb8), (12, ref._pb12), (16, ref._pb16), (20, ref._pb20),
        (24, ref._pb24), (28, ref._pb28), (32, ref._pb32), (36, ref._pb36), (40, ref._pb40),
        (44, ref._pb44), (48, ref._pb48), (52, ref._pb52), (56, ref._pb56), (60, ref._pb60),
        (64, ref._pb64), (68, ref._pb68), (72, ref._pb72), (76, ref._pb76), (80, ref._pb80),
        (84, ref._pb84), (88, ref._pb88), (96, ref._pb96), (100, ref._pb100)
    ]


def test_pb92_not_implemented_error():
    with pytest.raises(NotImplementedError):
        ref._pb92()


def test_pbX_dtype(supported_num):
    for num_fn in supported_num:
        num = num_fn[0]
        matrix = num_fn[1]()
        assert isinstance(matrix, np.ndarray), \
            f"dtype of retrun var expected np.ndarray, got {type(matrix)} in _pb{num}"


def test_pbX_shape(supported_num):
    for num_fn in supported_num:
        num = num_fn[0]
        matrix = num_fn[1]()
        assert matrix.shape == (num, num - 1), \
            f"shape of retrun var expected (X, X-1) for _pbX, got {matrix.shape} in _pb{num}"


def test_pbX_conversion_to_hadamard(supported_num):
    for num_fn in supported_num:
        num = num_fn[0]
        pb = num_fn[1]()
        hadamard = np.concatenate([pb, np.ones((num, 1))], axis=1)
        assert ((pb == 1) | (pb == -1)).all(), \
            f"all the elements in exmatrix should be either -1 or 1, got {pb} in _pb{num}"
        assert (hadamard @ hadamard.T == num * np.identity(num)).all(), \
            f"return var expected to be submatrix of hadamard matrix, got {pb} in _pb{num}"
