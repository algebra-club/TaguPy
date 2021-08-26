"""
_Generator Class of FullFactorial Design Generator Module
"""

import itertools
from typing import Iterable
import numpy as np
import numpy.matlib 


from tagupy.type import _Generator as Generator


class FullFact(Generator):
    '''
    Generator Class of FullFactorial Design Generator Module

    Method
    ------
    get_exmatrix(**info: Dict[str, Any]) -> np.ndarray
    get_alias_matrix(max_dim: int) -> np.ndarray

    Notes
    -----
    Full factorial design creates experimental points using all the possible combinations of the levels of the factors in each complete trial or replication of the experiments.  
    These experimental points are also called factorial points. 
    For three factors having four levels of each factor, considering full factorial design, total 4^3 (64) numbers of experiments have to be carried out. 
    If there are n replicates of complete experiments, then there will be n times of the single replication experiments to be conducted. 
    In the experimentation, it must have at least two replicates to determine a sum of squares due to error if all possible interactions are included in the model.
    
    Prasanta Sahoo, Tapan Kr. Barman, Woodhead Publishing Reviews, 2012,Pages 159-226,
    https://doi.org/10.1533/9780857095893.159.

    To assure the reliability of experiment, we
    reccomend you to replicate the same conditions
    and acquire multiple sets of the data.

    You can have the replicated FullFactorial Design Matrix
    at one time by setting n_rep as large
    non-zero natural number as you like.
    '''

    def __init__(self, n_rep: int) ->None:
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
        self.n_rep = n_rep
        assert isinstance(n_rep, int), \
            f"Error: n_rep expected int, got {type(n_rep)}"
        assert n_rep >= 1, \
            f"Error: e_rep expected integer >=1, got {n_rep}"
    
    def get_exmatrix(self, levels: Iterable[int]) -> np.array:
        '''
        create a full-factorial design

        Parameters
        ----------
        levels : List
            a list of integers which shows the number of level of each input factor
        
        Returns
        -------
        emat : numpy array(2d)
            the desing matrix with coded levels 0 to k-1 for a k-level factor

        Example
        -------
        >>> import tagupy
        >>> _model = tagupy.generator.FullFact(2)
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
        assert isinstance(levels, list), \
            f'Error: dtype of levels expected List \ngot{type(levels)}'
        for i in levels:
            assert isinstance(i, int), \
                f'Error: dtype of elements in the levels expected int \ngot {type(i)}'
            assert i >= 1, \
                f'Error: elements in the levels expected integer >=1 got {i}'

        levels_list = []
        for i in levels:
            levels_list.append([k for k in range(i)])
        emat = np.array(list((itertools.product(*levels_list))))
        emat = np.matlib.repmat(emat, self.n_rep, 1)

        return emat  




