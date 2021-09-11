import numpy as np
from tagupy.type import _Analyzer as Analyzer
from tagupy.utils import is_positive_int, is_int_2d_array
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

    def __init__(self, max_dim: int = 1):
        '''
        Parameters
        ----------
        max_dim: int
            number of dimensions for interaction;
            MET provides the results of only single interaction effect
            (equivalent to main effects of the given factors).
            max_dim recieves only integer value `1`.
        '''
        assert is_positive_int(max_dim), \
            f'max_dim: number of dimension expected positive integer,\
                got {max_dim}'

        assert max_dim == 1, \
            f'max_dim expected only integer value `1` \
                , for MET deals with main effect (max_dim = 1) and \
                    ignores higher dimensional interactions. Got {max_dim}'

        self.max_dim = max_dim

    def analyze(
        self,
        exmatrix: np.ndarray,
        resmatrix: np.ndarray,
    ):
        '''
        Parameters
        ----------
        `exmatrix`: numpy.ndarray
            `exmatrix` represents experiment matrix that describes experimental conditions
            for each run.

        `resmatrix`: numpy.ndarray
            `resmatrix` represents result matrix that describes experimental data
            that obtained through the experiments.

        Return
        ------
        AnalysisResult
            AnalysisResult includes three data: `exmatrix`, `resmatrix`, and `effectmatrix`.
            `exmatrix` and `resmatrix` are same as the each input values.
            `effectmatrix` is the result of analyzing the main effects for the two inputs.


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

        assert isinstance(resmatrix, np.ndarray), \
            f'resmatrix expected a numpy.ndarray, got: {type(resmatrix)}'

        assert is_int_2d_array(exmatrix), \
            f'exmatrix expected a matrix in which all elements are 0 or 1. \nGot: {exmatrix}'

        norm_exmatrix = exmatrix / exmatrix.sum(axis=0)
        centering_resmatrix = resmatrix - resmatrix.mean(axis=0)

        effectmatrix = norm_exmatrix.T @ centering_resmatrix

        return METResult(
            exmatrix=exmatrix,
            resmatrix=resmatrix,
            effectmatrix=effectmatrix,
        )
