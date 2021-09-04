import numpy as np
from math import factorial

from tagupy.type import _Simulator as Simulator

def combinations(n, r):
    return factorial(n) // (factorial(n - r) * factorial(r))

class Multinomial(Simulator):
    
    def __init__(self, n_factor: int, n_out: int, max_dim: int):
        assert type(n_factor)==int and n_factor>0, \
        'type of n_factor must be a natural number or 0'
        assert type(n_out)==int and n_out>0, \
        'type of n_out must be a natural number or 0' 
        assert type(max_dim)==int and max_dim>0, \
        'type of max_dim must be a natural number or 0'
        assert max_dim<=n_factor, \
        'Expected max_dim is equal or smaller than n_factor : Got larger max_dim'
        
        self.n_factor = n_factor
        self.n_out = n_out
        self.max_dim = max_dim
        
        temp = 0
        for i in range(max_dim):
            temp += combinations(n_factor, i)
        
        self.coef_column = temp
        self.coef_table = np.random.randn(n_out, self.coef_column)
        # self.var_err = np.random. <- errorのvarianceはどうするか？変動係数を一定にするようにするとか
                
    def simulate(self, exmatrix: np.ndarray) -> np.ndarray:
        
        assert type(exmatrix) == np.ndarray,\
        'AttributeError: The type of input matrix must be np.ndarray' 
        assert exmatrix.shape[1]==self.coef_table.shape[1],\
        f'ValueError: The dimention of exmatrix[1] must be n_factor, expect {self.coef_table.shape[1]}, got {exmatrix.shape[1]}'
        
        # err_mat = random.gauss()
        resmatrix = exmatrix @ self.coef_table.T
        
        return resmatrix