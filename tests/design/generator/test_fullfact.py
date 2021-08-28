import numpy as np
import pytest


from tagupy.design.generator import FullFact

'''
確認すべき項目
1. 上手く走るgreen
    1. instanceが作成できて__init__で作った変数を呼び出せるか
    2. 計画行列を出力できるか
        1. 実験計画表の形が正しい
            1. idx: 実験数: inputの要素の積＊n_rep
            2. col: len(levels)
        2. duplicationが存在しない
        3. 列毎に出ている値が合っている(uniqueで確認)
2. エラーが出るred
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
        assert isinstance(FullFact(i).n_rep, int), \
            f'Error: dtype of self.n_rep expected int,\
                 got {type(FullFact(i).n_rep)}'


def test_get_exmatrix_valid_output(correct_input):
    _model = FullFact(3)

    for i in correct_input:
        cor_shape = (np.prod(i) * _model.n_rep, len(i))
        ret_shape = _model.get_exmatrix(i).shape
        assert cor_shape == ret_shape,\
            f'Error: shape not matched,\
        expected: got: {_model.get_exmatrix(i).shape}'
        cor_len = len(_model.get_exmatrix(i))
        ret_len = len(np.unique(_model.get_exmatrix(i), axis=0)) * _model.n_rep
        assert cor_len == ret_len,\
            f'Error: Array has duplicated rows,\
        expected: {cor_len} unique rows, got: {ret_len}'
        for idx, j in enumerate(i):
            cor_ele = list(range(j))
            ret_ele = np.unique(_model.get_exmatrix(i)[:, idx])
            assert np.array_equal(cor_ele, ret_ele),\
                f'Error: column has unexpected elements,\
                column{idx} expected: {cor_ele}, got: {ret_ele}'


def test_getexmatrix_invalid_input():
    arg = ['a', 3.2, -1, None, [2], np.zeros((2, 3)), [0, 2, 3], [-1, 2, 3]]
    _model = FullFact(3)
    with pytest.raises(AssertionError):
        for i in arg:
            _model.get_exmatrix(i)


def test_getexmatrix_invalid_output(correct_input):
    _model = FullFact(3)
    for i in correct_input:
        assert isinstance(
            _model.get_exmatrix(i),
            np.ndarray
        ), \
            f'Error: dtype of ematrix expected np.adarray,\
                got {type(_model.get_exmatrix(correct_input))}'
