import numpy as np
from math import factorial
import pytest

from tagupy.simulator import Multinomial

def combinations(n, r):
    return factorial(n) // (factorial(n - r) * factorial(r))

@pytest.fixture

def correct_input():
    return [1, 2, 3, 4, 5, 6, 30]


# Red test: inappropriate inputs : n_factor
def test_invalid_n_factor(): # 何のテストなのかわかりやすい関数名
    arg = [0, -1, 2.0, np.zeros(5), 'a', True, [1, 2]]
    n_out = 2
    max_dim = 2
    
    for v in arg:
        with pytest.raises(AssertionError) as e:
            Multinomial(v, n_out, max_dim)
            assert f'{v}' in f'{e.value}', f'NoReasons: Inform the AssertionError reasons, got {e.value}'

# Green test
def test_correct_n_factor(correct_input):
    n_out = 2
    max_dim = 2
    
    for i, v in enumerate(correct_input):
        
        temp = 0
        for j in range(max_dim):
            temp += combinations(v, j)
        ref_col = temp
        
        assert Multinomial(v, n_out, max_dim).n_factor == v, \
        f'self.n_factor expected {v}, got {Multinomial(v, n_out, max_dim).n_factor}'
        
        assert Multinomial(v, n_out, max_dim).coef_column == ref_col,\
        f'shape of self.coef_table expected {n_out, ref_col}, got {Multinomial(v, n_out, max_dim).coef_table.shape[0], Multinomial(v, n_out, max_dim).coef_column}'



# Red test: inappropriate inputs : n_factor
def test_invalid_n_factor(): # 何のテストなのかわかりやすい関数名
    arg = [0, -1, 2.0, np.zeros(5), 'a', True, [1, 2]]
    n_out = 2
    max_dim = 2
    
    for v in arg:
        with pytest.raises(AssertionError) as e:
            Multinomial(v, n_out, max_dim)
            assert f'{v}' in f'{e.value}', f'NoReasons: Inform the AssertionError reasons, got {e.value}'

# Green test
def test_correct_n_factor(correct_input):
    n_out = 2
    max_dim = 2
    ref_col = []
 
    for i, v in enumerate(correct_input):
        
        temp = 0
        for j in range(max_dim):
            temp += combinations(v, j)
        ref_col = temp
        
        assert Multinomial(v, n_out, max_dim).n_factor == v, \
        f'self.n_factor expected {v}, got {Multinomial(v, n_out, max_dim).n_factor}'
        
        assert Multinomial(v, n_out, max_dim).coef_column == ref_col,\
        f'shape of self.coef_table expected {n_out, ref_col}, got {Multinomial(v, n_out, max_dim).coef_table.shape[0], Multinomial(v, n_out, max_dim).coef_column}'

# Red test: inappropriate inputs : n_out
def test_invalid_n_out(): # 何のテストなのかわかりやすい関数名
    arg = [0, -1, 2.0, np.zeros(5), 'a', True, [1, 2]]
    n_factor = 2
    max_dim = 2
    
    for v in arg:
        with pytest.raises(AssertionError) as e:
            Multinomial(n_factor, v, max_dim)
            assert f'{v}' in f'{e.value}', f'NoReasons: Inform the AssertionError reasons, got {e.value}'

# Green test
def test_correct_n_out(correct_input):
    n_factor = 2
    max_dim = 2
    
    for i, v in enumerate(correct_input):
        assert Multinomial(n_factor, v, max_dim).n_out == v, \
        f'self.n_out expected {v}, got {Multinomial(n_factor, v, max_dim).n_out}'
        
        assert Multinomial(n_factor, v, max_dim).coef_table.shape[0] == v,\
        f'shape of self.coef_table expected {v, Multinomial(n_factor, v, max_dim).coef_column}, got {Multinomial(n_factor, v, max_dim).coef_table.shape}'


# Red test: inappropriate inputs : max_dim
def test_invalid_max_dim():
    arg = [0, -1, 2.0, np.zeros(5), 'a', True, [1, 2], 10] # n_factor<max_dimを追加
    n_factor = 2
    n_out = 2
    
    for v in arg:
        with pytest.raises(AssertionError) as e:
            Multinomial(n_factor, n_out, arg)
            assert f'{v}' in f'{e.value}', f'NoReasons: Inform the AssertionError reasons, got {e.value}'

# Green test
def test_correct_max_dim():
    n_factor = 10
    n_out = 2
    arg = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for i, v in enumerate(arg):
        assert Multinomial(n_factor, n_out, v).max_dim == arg[i], \
        f'self.max_dim expected {arg[i]}, got {Multinomial(n_factor, n_out, v).max_dim}'
        
        # !!!のちのちcombination が加味されるので
        temp = 0
        for j in range(v):
            temp += combinations(n_factor, j)
        ref_col = temp

        assert Multinomial(n_factor, n_out, v).coef_column == ref_col,\
        f'shape of self.coef_table expected {n_out}, {ref_col}, got {Multinomial(n_factor, n_out, v).coef_table.shape}'


# Red test: inappropriate inputs : exmat
def test_invalid_exmat():
    arg1 = [1, 0, -1, 2.0, 'a', True, [1, 2]] 
    arg2 = [np.arange(15).reshape(3, 5), np.zeros(6).reshape(2, 3), np.random.random((1, 8))] # ex_mat.shape[1] != n_factor:2
    n_factor = 2
    n_out = 2
    max_dim = 2
    test = Multinomial(n_factor, n_out, max_dim)
    
    for v1 in arg1:
        with pytest.raises(AssertionError) as e:
            test.simulate(v1)
            assert f'{v1}' in f'{e.value}', f'NoReasons: Inform the AssertionError reasons, got {e.value}'
    
    for v2 in arg2:
        with pytest.raises(AssertionError) as e:
            test.simulate(v2)
            assert f'{v2}' in f'{e.value}', f'NoReasons: Inform the AssertionError reasons, the shape of input {e.value} was not equal to {n_factor}'

# Green test
def test_correct_exmat():
    n_factor = [5, 4, 3, 8]
    n_out = 2
    max_dim = 2
    arg = [np.arange(10).reshape(2, 5), np.zeros(20).reshape(5, 4), np.ones(9).reshape(3, 3), np.random.random(8).reshape(1,8)] 
    
    for i, v in enumerate(arg):
        test = Multinomial(n_factor[i], n_out, max_dim)
        assert test.simulate(v).exmatrix == arg[i], \
        f'ex_matrix expected {arg[i]}, got {test.simulate(v).exmatrix}'

# Green test: output
# datatype / shape / contents 

def test_valid_resmat():
    
    n_factor = [3, 4, 5, 8]
    n_out = 2
    max_dim = 2
    ex_mat = [np.ones((3,3)), np.zeros((5,4)), np.arange(10).reshape(2, 5), np.random.random((1, 8))] 
    ref_col = np.zeros(n_factor)
    '''
    for i in n_factor:
        temp = 0
        for j in max_dim:
            temp += combinations(i, j)
        ref_col[i] = temp
    '''
    
    for i, v in enumerate(ex_mat):
        test = Multinomial(n_factor[i], n_out, max_dim)
        ref = test.simulate(v)
        res = v @ self.coef_table.T
        
        # shape
        assert res.shape == ref[i].shape, \
        f'shape of result_matrix expected {ref[i].shape}, got {res.shape}'
        
        # datatype
        assert isinstance(ref[i], type(res)), \
        f'type of result_matrix expected {type(ref[i])}, got {type(res)}'
        
        # culculation
        assert res == ref[i], \
        f'result_matrix expected {ref[i]}, got{res}'