"""
Utility functions
"""
from itertools import combinations
from typing import List

import numpy as np


__all__ = [
    "get_corr_matrix",
    "get_comb_name",
]


def get_corr_matrix(exmatrix: np.ndarray, max_dim: int) -> np.ndarray:
    """
    Return Correlation Matrix

    Parameters
    ----------
    max_dim: int
        maximum dimension in Correlation Matrix

    Returns
    -------
    correlation matrix: numpy.ndarray
        Correlation Matrix, if you want more details, see also Notes.

    Notes
    -----
    https://www.jmp.com/support/help/en/16.1/index.shtml#page/jmp/color-map-on-correlations.shtml

    Example
    -------
    >>> from tagupy.design import generator
    >>> from tagupy.utils import get_corr_matrix
    >>> model = generator.OneHot(n_rep=1)
    >>> exmatrix = model.get_exmatrix(n_factor=3)
    >>> get_corr_matrix(exmatrix, max_dim=1)
    array([[ 1.        , -0.33333333, -0.33333333],
           [-0.33333333,  1.        , -0.33333333],
           [-0.33333333, -0.33333333,  1.        ]])
    >>> # You can only set max_dim = 1
    >>> get_corr_matrix(exmatrix, max_dim=2)
    Traceback (most recent call last):
      ...
    NotImplementedError: Support only max_dim = 1
    """
    # TODO: Check the if the definition is right
    # TODO: Generalize max_dim
    pass
    if max_dim != 1:
        raise NotImplementedError('Support only max_dim = 1')

    return np.corrcoef(exmatrix.T)


def get_comb_name(
    factor_name: List[str],
    max_dim: int,
    operator: str = ''
) -> List[str]:
    '''
    Get the combination string from string list
    '''
    _factor_multi_name = [list(combinations(factor_name, d)) for d in range(1, max_dim+1)]
    factor_multi_name = sum(_factor_multi_name, [])

    return [f'{operator}'.join(f) for f in factor_multi_name]
