import numpy as np
from math import factorial
import pytest

from tagupy.simulator import Multinomial
from tagupy.utils import get_comb_name


def get_n_comb(n, r):
    return factorial(n) // (factorial(n - r) * factorial(r))


@pytest.fixture
def correct_input():
    return [1, 2, 3, 4, 5, 6, 30]


# Red test: inappropriate in
def test_invalid_n_factor():
    arg = [0, -1, 2.0, np.zeros(5), 'a', True, [1, 2]]
    n_out = 2
    max_dim = 2

    for v in arg:
        with pytest.raises(ValueError) as e:
            Multinomial(v, n_out, max_dim)
        assert f'{v}' in f'{e.value}', \
            f'NoReasons: Inform the AssertionError reasons, got {e.value}'


# Green test
def test_correct_n_factor(correct_input):
    n_out = 2
    max_dim = 2

    for v in correct_input[1:]:
        ref_col = get_comb_name([f'x{i}' for i in range(v)], max_dim)
        res = Multinomial(v, n_out, max_dim)

        assert res.n_factor == v, \
            f'self.n_factor expected {v}, got {res.n_factor}'

        assert res.coef_column == ref_col,\
            f'shape of self.coef_table expected {(n_out, ref_col)}, got {res.coef_table.shape}'


# Red test: inappropriate inputs : n_out
def test_invalid_n_out():
    arg = [0, -1, 2.0, np.zeros(5), 'a', True, [1, 2]]
    n_factor = 2
    max_dim = 2

    for v in arg:
        with pytest.raises(ValueError) as e:
            Multinomial(n_factor, v, max_dim)
        assert f'{v}' in f'{e.value}',\
            f'NoReasons: Inform the AssertionError reasons, got {e.value}'


# Green test
def test_correct_n_out(correct_input):
    n_factor = 2
    max_dim = 2

    for v in correct_input:
        res = Multinomial(n_factor, v, max_dim)
        assert res.n_out == v, \
            f'self.n_out expected {v}, got {res.n_out}'

        assert res.coef_table.shape[0] == v,\
            f'shape of self.coef_table expected {(v, res.coef_table.shape[1])}, got {res.coef_table.shape}'  # noqa: E501


# Red test: inappropriate inputs : max_dim
def test_invalid_max_dim():
    # n_factor<max_dimを追加
    arg = [0, -1, 2.0, np.zeros(5), 'a', True, [1, 2], 10]
    n_factor = 2
    n_out = 2

    for v in arg:
        try:
            res = Multinomial(n_factor, n_out, v)
        except ValueError as e:
            assert f'{v}' in f'{e.args[0]}',\
                f'NoReasons: Inform the AssertionError reasons, got {e.args[0]}'
        except Exception as e:
            pytest.fail(f'{type(e)}: {e.args}')
        else:
            pytest.fail(f'Did not Raise ValueError, got {v} -> {res}')


# Green test
def test_correct_max_dim():
    n_factor = 10
    n_out = 2
    arg = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i, v in enumerate(arg):
        res = Multinomial(n_factor, n_out, v)
        assert res.max_dim == arg[i], \
            f'self.max_dim expected {arg[i]}, got {res.max_dim}'

        # !!!のちのちcombination が加味されるので
        factor = [f'x{i}' for i in range(n_factor)]
        ref_col = get_comb_name(factor, v)

        assert res.coef_column == ref_col,\
            f'shape of self.coef_table expected {n_out}, {ref_col}, got {res.coef_table.shape}'


# Red test: inappropriate inputs : exmat
def test_invalid_exmat():
    arg1 = [1, 0, -1, 2.0, 'a', True, [1, 2]]
    # ex_mat.shape[1] != n_factor:2
    arg2 = [np.arange(15).reshape(3, 5), np.zeros((2, 3)), np.random.random((1, 8))]
    n_factor = 2
    n_out = 2
    max_dim = 2
    test = Multinomial(n_factor, n_out, max_dim)

    for v1 in arg1:
        with pytest.raises(ValueError) as e:
            test.simulate(v1)
            assert f'{v1}' in f'{e.value}',\
                f'NoReasons: Inform the AssertionError reasons, got {e.value}'

    for v2 in arg2:
        with pytest.raises(ValueError) as e:
            test.simulate(v2)
            assert f'{v2}' in f'{e.value}',\
                f'NoReasons: Inform the AssertionError reasons'


# Green test
def test_correct_exmat():
    n_factor = [5, 4, 3, 8]
    n_out = 2
    max_dim = 2
    args = [
        np.arange(10).reshape(2, 5),
        np.zeros((5, 4)),
        np.ones(9).reshape(3, 3),
        np.random.random(8).reshape(1, 8),
    ]

    for f, arg in zip(n_factor, args):
        Multinomial(f, n_out, max_dim).simulate(arg)


# Green test: output
# datatype / shape / contents
def test_valid_resmat():

    n_factor = [3, 4, 5, 8]
    n_out = 2
    max_dim = 2
    ex_mat = [
        np.ones((3, 3)),
        np.zeros((5, 4)),
        np.arange(10).reshape(2, 5),
        np.random.random((1, 8))
    ]

    for i, v in enumerate(ex_mat):
        test = Multinomial(n_factor[i], n_out, max_dim)
        try:
            ref = test.simulate(v)
        except Exception as ex:
            pytest.fail(
                f'{v.shape, test.coef_table.T.shape, test.err.shape} \n {ex.args} \n {ex.__class__}'
            )

        res = v @ test.coef_table.T
        ref = test.simulate(v)

        # shape
        assert res.shape == ref[i].shape, \
            f'shape of result_matrix expected {ref[i].shape}, got {res.shape}'

        # datatype
        assert isinstance(ref[i], type(res)), \
            f'type of result_matrix expected {type(ref[i])}, got {type(res)}'

        # culculation
        assert res == ref[i], \
            f'result_matrix expected {ref[i]}, got{res}'
