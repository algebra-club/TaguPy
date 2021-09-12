import numpy as np
from numpy.typing import NDArray

from tagupy.type import _Simulator as Simulator
from tagupy.utils import is_positive_int, get_comb_name


class Multinomial(Simulator):
    def __init__(self, n_factor: int, n_out: int, max_dim: int):
        args = ['n_factor', 'n_out', 'max_dim']
        for idx, arg in enumerate((n_factor, n_out, max_dim)):
            if not is_positive_int(arg):
                raise ValueError(f"{args[idx]} must be a natural number or 0, got {arg}")

        if max_dim > n_factor:
            raise ValueError(f'expected max_dim ≦ n_factor: max_dim={max_dim}, n_factor={n_factor}')

        self.n_factor = n_factor
        self.n_out = n_out
        self.max_dim = max_dim

        factors = [f'x{i}' for i in range(n_factor)]
        self.coef_column = get_comb_name(factors, max_dim)
        self.coef_table = np.random.randn(n_out, len(self.coef_column))
        # TODO: errorのvarianceはどうするか？変動係数を一定にするようにするとか
        self.err = np.zeros_like(self.coef_table)

    def simulate(self, exmatrix: NDArray[np.integer]) -> NDArray[np.floating]:
        if type(exmatrix) != np.ndarray:
            raise ValueError(f'The type of input matrix must be np.ndarray, got {type(exmatrix)}')
        if exmatrix.shape[1] != self.n_factor:
            raise ValueError(f'Number of columns in exmatrix must be the same as n_factor\nexpect {self.n_factor}, got {exmatrix.shape[1]}')  # noqa: E501

        resmatrix = exmatrix @ self.coef_table.T + self.err
        return resmatrix
