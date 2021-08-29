import numpy as np
import pytest


from tagupy.design.generator import FullFact


@pytest.fixture
def correct_input():
    arg = [[2, 2, 3], [3, 3, 3, 3, 3], [5, 10, 15], [20, 20], [2]]
    return arg


def test_init_vaild_input():
    arg = [1, 2, 10]
    exp = [1, 2, 10]
    for i, v in enumerate(arg):
        ret = FullFact(v)
        assert ret.n_rep == exp[i], \
            f'Error: invalid instance-var n_rep > expected {exp[i]}, got {ret.n_rep}'


def test_init_invalid_input():
    arg = ['a', 3.2, None, [2], np.zeros((2, 3)), np.inf, -np.inf, 0, -1]

    for i in arg:
        with pytest.raises(AssertionError) as e:
            FullFact(i)
        assert f'{i}' in f"{e.value}", \
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_init_valid_output():
    arg = [1, 2, 3, 10]
    for i in arg:
        rep = FullFact(i).n_rep
        assert rep == i, \
            f'Error: invalid dtype > expected {i}, got {rep}'


def test_get_exmatrix_valid_output(correct_input):
    model = FullFact(3)

    for i in correct_input:
        cor_shape = (np.prod(i) * model.n_rep, len(i))
        ret_shape = model.get_exmatrix(i).shape
        assert cor_shape == ret_shape,\
            f'Error: shape not matched > expected: {cor_shape}, got: {ret_shape}'

        cor_len = len(model.get_exmatrix(i))
        ret_len = len(np.unique(model.get_exmatrix(i), axis=0)) * model.n_rep
        assert cor_len == ret_len,\
            f'Error: Array has duplicated rows > expected: {cor_len} unique rows, got: {ret_len}'

        for idx, j in enumerate(i):
            cor_ele = list(range(j))
            ret_ele = np.unique(model.get_exmatrix(i)[:, idx])
            assert np.array_equal(cor_ele, ret_ele),\
                f'Error: Unexpected elements in Column[{idx}] > expected: {cor_ele}, got: {ret_ele}'


def test_getexmatrix_invalid_input():
    arg = [
        ['a', 3.2, -1, None, np.zeros((2, 3))],
        [['a'], ['e', 2, 3]],
        [[0, 2, 3], [-1, 2, 3]]
        ]
    model = FullFact(3)

    with pytest.raises(AssertionError) as e:
        for idx in range(len(arg)):
            for i in arg[idx]:
                model.get_exmatrix(i)
                assert f'{i}' in f"{e.value}",\
                    f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_getexmatrix_invalid_output(correct_input):
    model = FullFact(3)
    for i in correct_input:
        ret = model.get_exmatrix(i)
        assert isinstance(ret, np.ndarray), \
            f'Error: dtype of ematrix expected np.adarray, got {type(ret)}'
