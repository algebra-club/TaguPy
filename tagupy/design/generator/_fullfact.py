"""
_Generator Class of FullFactorial Design Generator Module
"""
import numpy as np

from typing import Iterable

from tagupy.type import _Generator as Generator
from tagupy.utils import is_positive_int, is_positive_int_list


class FullFact(Generator):
    '''
    Generator Class of FullFactorial Design Generator Module

    Method
    ------
    get_exmatrix(self, levels: Iterable[int]) -> np.ndarray

    Notes
    -----
    Full factorial design creates all the possible combinations of the levels by each factor.
    This design takes n_factors^m_levels experiments if levels are the same numbers.
    When number of replications (n_rep) is set as non-zero natural number,
      there will be n times of the single replication experiments.
    It is recommended to have at least 2 replicates to determine a sum of squares due to error.

    see also:
    Prasanta Sahoo, Tapan Kr. Barman, Woodhead Publishing Reviews, 2012,Pages 159-226,
    https://doi.org/10.1533/9780857095893.159.
    '''

    def __init__(self, n_rep: int):
        '''
        Parameters
        ----------
        n_rep: int
            number of replications; that value is applied
            when the whole set of experiment is replicated
            for the quality assurance of the experiment data.
            (when n_rep = 1, it implies that a single run
            for each condition will be planed)
        '''
        assert is_positive_int(n_rep), \
            f"Invalid input: n_rep expected positive (>0) integer, got {type(n_rep)}::{n_rep}"
        self.n_rep = n_rep

    def get_exmatrix(self, levels: Iterable[int]) -> np.ndarray:
        '''
        create a full-factorial design

        Parameters
        ----------
        levels : Iterable[int]
            a list of integers which shows the number of level of each input factor

        Returns
        -------
        exmatrix(: np.ndarray(n_experiments * n_factor)
            the experiment matrix with all the possible combinations of the levels by each factor.

        Note
        ----
        experiment design containig less than 10 factors is expected

        Example
        -------
        >>> from tagupy.design.generator import FullFact
        >>> _model = FullFact(n_rep=2)
        >>> _model.get_exmatrix([2,3,2])
        array([[0, 0, 0],
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
        '''
        assert is_positive_int_list(levels), \
            f'Invalid input: levels is List of positive (>0) integer, got {type(levels)}::{levels}'

        ary = [np.arange(i, dtype=np.int8) for i in levels]
        exmatrix = np.indices(list(levels)).reshape(len(ary), -1).T

        return np.vstack([exmatrix] * self.n_rep)
