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
        exmatrix: tuple[list[str], np.ndarray],
        result: tuple[list[str], np.ndarray],
    ):
        '''
        Parameters
        ----------
        exmatrix: tuple[list[str], np.ndarray]

        result: tuple[list[str], np.ndarray]

        Returns
        ----------
        report: tuple[list[str], np.ndarray]
        '''

        pre_result = result - result.mean(axis=0)
        pre_exmatrix = exmatrix/exmatrix.sum(axis=0)

        return pre_exmatrix.T@pre_result
