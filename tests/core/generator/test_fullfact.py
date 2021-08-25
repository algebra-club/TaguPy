import numpy as np
import pytest


from tagupy.core.generator import FullFact

'''
確認すべき項目
1. 上手く走る
    1. instanceが作成できて__init__で作った変数を呼び出せるか   
    2. 計画行列を出力できるか
2. エラーが出る
    1. __init__の作動
        1. 入力の型がおかしい
        2. 出力の型がおかしい
    2. 計画行列作成
        1. 入力の型がおかしい
            1. list形式で入力されない
            2. listが空である
            3. listの中身がintではない
            4. 水準が１以下の物がある
        2. 出力の型がおかしい
'''
@pytest.fixture
def correct_input():
    arg = [2, 3, 2]
    return arg


def test_init():
    arg = [1, 2, 10]
    exp = [1, 2, 10]
    for i, v in enumerate(arg):
        assert FullFact(v).n_rep == exp[i], \
            f'Error: self.n_rep expected {exp[i]}, \ngot {FullFact(v).n_rep}'


def test_get_exmatrix(correct_input):
    exp = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 2, 0],
                    [0, 2, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 2, 0],
                    [1, 2, 1],
                    [0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 2, 0],
                    [0, 2, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1],
                    [1, 2, 0],
                    [1, 2, 1]])
    _model = FullFact(2)
    np.testing.assert_array_equal(_model.get_exmatrix(correct_input), exp), \
        'Error: The method "get_exmatrix" did not output the expected matrix'


def test_init_invalid_input():
    arg = ['a', 3.2, -1, None, [2], np.zeros((2,3))]
    with pytest.raises(AssertionError):
        for i in arg:
            FullFact(i)


def test_init_invalid_output(correct_input):
    for i in correct_input:
        assert isinstance(FullFact(i).n_rep, int), \
            f'Error: dtype of self.n_rep expected int \ngot{type(FullFact(i).n_rep)}'
        

def test_getexmatrix_invalid_input():
    arg = ['a', 3.2, -1, None, [2], np.zeros((2,3)), [0, 2, 3], [-1, 2, 3]]
    _model = FullFact(3)
    with pytest.raises(AssertionError):
        for i in arg:
            _model.get_exmatrix(i)


def test_getexmatrix_invalid_output(correct_input):
    _model = FullFact(3)
    assert isinstance(
        _model.get_exmatrix(correct_input),
        np.ndarray
    ), \
        f'Error: dtype of ematrix expected np.adarray \ngot {type(_model.get_exmatrix(correct_input))}'
