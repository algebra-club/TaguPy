import numpy as np
from tagupy.type import _Analyzer as Analyzer
from typing import NamedTuple


class METResult(NamedTuple):
    exmatrix: np.ndarray
    resmatrix: np.ndarray
    effectmatrix: np.ndarray


class MET(Analyzer):
    '''
    Analyzer Class of main effect table

    Notes
    ----------
    This analyzer provides analyzing main effect table.
    You need two matrices: exmatrix and resmatrix.
    The exmatrix represents which factors to include or not.
    The resmatrix contains the results obtained by the experiment.

    When you execute this analyzer, it return the analysis result instance.
    See also: /tagupy/type/_analysis_result.py

    '''

    def __init__(self, n_dim: int):
        '''
        Parameters
        ----------
        n_dim: int
            number of dimensions for interaction; MET provides the results of only single interaction effect.
            n_dim recieves only `1`.
        '''
        assert n_dim == 1, f'n_dim expected only `1`, got {n_dim}'

        self.n_dim = n_dim

    def analyze(
        self,
        exmatrix: np.ndarray,
        resmatrix: np.ndarray,
    ):
        '''
        Parameters
        ----------
        exmatrix: np.ndarray

        resmatrix: np.ndarray


        Return
        ----------
        AnalysisResult


        Example
        ----------
        >>> import numpy as np
        >>> from tagupy.design.analyzer import MET
        >>> analyzer = MET()
        >>> result = analyzer.analyze(
        ...     np.array([[1, 1, 0, 1],
        ...               [1, 1, 1, 0],
        ...               [1, 0, 1, 1],
        ...               [1, 0, 0, 0]]),
        ...     np.array([[3, 4, 7],
        ...               [4, 9, 8],
        ...               [5, 8, 3],
        ...               [8, 3, 6]]),
        ... )

        >>> result.exmatrix
        array([[1, 1, 0, 1],
               [1, 1, 1, 0],
               [1, 0, 1, 1],
               [1, 0, 0, 0]])

        >>> result.resmatrix
        array([[3, 4, 7],
               [4, 9, 8],
               [5, 8, 3],
               [8, 3, 6]])

        >>> result.effectmatrix
        array([[ 0. ,  0. ,  0. ],
               [-1.5,  0.5,  1.5],
               [-0.5,  2.5, -0.5],
               [-1. ,  0. , -1. ]])
        '''

        assert isinstance(exmatrix, np.ndarray), \
            f'exmatrix expected a np.ndarray, got: {type(exmatrix)}'

        assert isinstance(resmatrix, np.ndarray), \
            f'resmatrix expected a np.ndarray, got: {type(resmatrix)}'

        assert np.logical_or(exmatrix == 0, exmatrix == 1).all(), \
            f'exmatrix expected a matrix in which all elements are 0 or 1. \nGot: {np.full(exmatrix)}'

        pre_exmatrix = exmatrix / exmatrix.sum(axis=0)
        pre_resmatrix = resmatrix - resmatrix.mean(axis=0)

        effectmatrix = pre_exmatrix.T @ pre_resmatrix

        return METResult(
            exmatrix=exmatrix,
            resmatrix=resmatrix,
            effectmatrix=effectmatrix,
        )
