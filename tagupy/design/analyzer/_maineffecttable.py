import numpy as np
from tagupy.type import _Analyzer as Analyzer


class MainEffectTable(Analyzer):
    def __init__(self):
        '''
        Method
        ----------
            analyze(exmatrix: np.ndarray) -> AnalysisResult
        '''

    def analyze(
        self,
        exmatrix: np.ndarray,
        result: np.ndarray,
    ):
        '''
        Parameters
        ----------
        exmatrix: np.ndarray

        result: np.ndarray

        Return
        ----------
        report: np.ndarray

        Example
        ----------
        >>> import numpy as np
        >>> from tagupy.design.analyzer import MainEffectTable
        >>> analyzer = MainEffectTable()
        >>> analyzer.analyze(
        ...     np.array([[1, 1, 0, 1],
        ...               [1, 1, 1, 0],
        ...               [1, 0, 1, 1],
        ...               [1, 0, 0, 0]]),
        ...     np.array([[3, 4, 7],
        ...               [4, 9, 8],
        ...               [5, 8, 3],
        ...               [8, 3, 6]]),
        ... )
        array([[ 0. ,  0. ,  0. ],
               [-1.5,  0.5,  1.5],
               [-0.5,  2.5, -0.5],
               [-1. ,  0. , -1. ]])
        '''

        pre_result = result - result.mean(axis=0)
        pre_exmatrix = exmatrix / exmatrix.sum(axis=0)

        return pre_exmatrix.T @ pre_result
