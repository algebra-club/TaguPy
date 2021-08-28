import numpy as np
import pytest


from tagupy.design.generator import FullFact


@pytest.fixture
def correct_input():
    arg = [[2, 2, 3], [3, 3, 3, 3, 3], [5, 10, 15], [20, 20]]
    return arg


def test_init_vaild_input():
    arg = [1, 2, 10]
    exp = [1, 2, 10]
    for i, v in enumerate(arg):
        assert FullFact(v).n_rep == exp[i], \
            f'Error: self.n_rep expected {exp[i]}, got {FullFact(v).n_rep}'


def test_init_invalid_input():
    arg = ['a', 3.2, -1, None, [2], np.zeros((2, 3))]
    with pytest.raises(AssertionError):
        for i in arg:
            FullFact(i)


def test_init_invalid_output():
    arg = [1, 2, 3, 10]
    for i in arg:
        rep = FullFact(i).n_rep
        assert isinstance(rep, int), \
            f'Error: dtype of self.n_rep expected int,\
                 got {type(rep)}'


def test_get_exmatrix_valid_output(correct_input):
    model = FullFact(3)

    for i in correct_input:
        cor_shape = (np.prod(i) * model.n_rep, len(i))
        ret_shape = model.get_exmatrix(i).shape
        assert cor_shape == ret_shape,\
            f'Error: shape not matched,\
        expected: {cor_shape}got: {ret_shape}'
        cor_len = len(model.get_exmatrix(i))
        ret_len = len(np.unique(model.get_exmatrix(i), axis=0)) * model.n_rep
        assert cor_len == ret_len,\
            f'Error: Array has duplicated rows,\
        expected: {cor_len} unique rows, got: {ret_len}'
        for idx, j in enumerate(i):
            cor_ele = list(range(j))
            ret_ele = np.unique(model.get_exmatrix(i)[:, idx])
            assert np.array_equal(cor_ele, ret_ele),\
                f'Error: column has unexpected elements,\
                column{idx}, expected: {cor_ele}, got: {ret_ele}'


def test_getexmatrix_invalid_input():
    arg = ['a', 3.2, -1, None, [2], np.zeros((2, 3)), [0, 2, 3], [-1, 2, 3]]
    model = FullFact(3)
    with pytest.raises(AssertionError):
        for i in arg:
            model.get_exmatrix(i)


def test_getexmatrix_invalid_output(correct_input):
    model = FullFact(3)
    for i in correct_input:
        assert isinstance(
            model.get_exmatrix(i),
            np.ndarray
        ), \
            f'Error: dtype of ematrix expected np.adarray,\
                got {type(model.get_exmatrix(i))}'
