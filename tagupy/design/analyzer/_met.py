import numpy as np
from tagupy.type import _Analyzer as Analyzer
from tagupy.utils import is_positive_int
from typing import NamedTuple


class METResult(NamedTuple):
    exmatrix: np.ndarray
    resmatrix: np.ndarray
    effectmatrix: np.ndarray


class MET(Analyzer):
    '''
    Analyzer Class of main effect table (MET)

    Notes
    -----
    This model can be used for data analysis on main effects of each factor, and you'll get
    the result in table format (we call it Main Effect Table).
    You need two matrices: `exmatrix` and `resmatrix`.
    `exmatrix` represents experiment matrix that describes experimental conditions for each run.
    `resmatrix` represents result matrix that describes experimental data that obtained through
    the experiments.

    When you execute this analyzer, it return the analysis result created by `NamedTuple`.
    This result includes three contents: `exmatrix`, `resmatris`, and `effectmatrix`.
    `exmatrix` and `resmatrix` are the same as each own input when execute analyze method.
    `effectmatrix` is the result of analyzing the main effects for two inputs.

    '''

    def __init__(self, max_dim_interaction: int = 1):
        '''
        Parameters
        ----------
        max_dim_interaction: int
            number of dimensions for interaction;
            MET provides the results of only single interaction effect
            (equivalent to main effects of the given factors).
            max_dim_interaction recieves only integer value `1`.
        '''
        assert is_positive_int(max_dim_interaction), \
            f'max_dim_interaction: number of dimension expected positive integer, got {max_dim_interaction}'

        assert max_dim_interaction == 1, \
            f'max_dim_interaction expected only integer value `1` \
                , for MET deals with main effect (max_dim_interaction = 1) and \
                    ignores higher dimensional interactions. Got {max_dim_interaction}'

        self.max_dim_interaction = max_dim_interaction

    def analyze(
        self,
        exmatrix: np.ndarray,
        resmatrix: np.ndarray,
    ):
        '''
        Parameters
        ----------
        exmatrix: numpy.ndarray

        resmatrix: numpy.ndarray


        Return
        ------
        AnalysisResult


        Example
        -------
        >>> import numpy as np
        >>> from tagupy.design.analyzer import MET
        >>> analyzer = MET(1)
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
            f'exmatrix expected a numpy.ndarray, got: {type(exmatrix)}'

        assert isinstance(resmatrix, np.ndarray), \
            f'resmatrix expected a numpy.ndarray, got: {type(resmatrix)}'

        assert np.logical_or(exmatrix == 0, exmatrix == 1).all(), \
            f'exmatrix expected a matrix in which all elements are 0 or 1. \nGot: {exmatrix}'

        pre_exmatrix = exmatrix / exmatrix.sum(axis=0)
        pre_resmatrix = resmatrix - resmatrix.mean(axis=0)

        effectmatrix = pre_exmatrix.T @ pre_resmatrix

        return METResult(
            exmatrix=exmatrix,
            resmatrix=resmatrix,
            effectmatrix=effectmatrix,
        )
