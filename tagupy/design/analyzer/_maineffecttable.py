import numpy as np
from tagupy.type import _Analyzer as Analyzer
from typing import NamedTuple

from tagupy.type import _AnalysisResult as AnalysisResult


class MainEffectTable(Analyzer):
    '''
    Analyzer Class of MainEffectTable

    Notes
    ----------
    This analyzer provides analyzing main effect table.
    You need two matrices: exmatrix and resmatrix.
    The exmatrix represents which factors to include or not.
    The resmatrix contains the results obtained by the experiment.

    When you execute this analyzer, it return the analysis result instance.
    See also: /tagupy/type/_analysis_result.py

    '''

    def __init__(self):
        pass

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
        AnalysisResult(exmatrix=array([[1, 1, 0, 1],
               [1, 1, 1, 0],
               [1, 0, 1, 1],
               [1, 0, 0, 0]]), effectmatrix=array([[ 0. ,  0. ,  0. ],
               [-1.5,  0.5,  1.5],
               [-0.5,  2.5, -0.5],
               [-1. ,  0. , -1. ]]), resmatrix=array([[3, 4, 7],
               [4, 9, 8],
               [5, 8, 3],
               [8, 3, 6]]))
        '''

        assert isinstance(exmatrix, np.ndarray), \
            f'exmatrix expected a np.ndarray, got: {type(exmatrix)}'

        assert isinstance(resmatrix, np.ndarray), \
            f'resmatrix expected a np.ndarray, got: {type(resmatrix)}'

        assert np.logical_or(exmatrix == 0, exmatrix == 1).all(), \
            f'exmatrix expected a matrix in which all elements are 0 or 1. \nGot: {exmatrix}'

        pre_exmatrix = exmatrix / exmatrix.sum(axis=0)
        pre_resmatrix = resmatrix - resmatrix.mean(axis=0)

        effectmatrix = pre_exmatrix.T @ pre_resmatrix

        return METAnalysisResult(
            exmatrix=exmatrix,
            resmatrix=resmatrix,
            effectmatrix=effectmatrix,
        )


class METNamedTuple(NamedTuple):
    effectmatrix: np.ndarray


class METAnalysisResult(AnalysisResult):
    def __init__(self, exmatrix, resmatrix, effectmatrix):
        self._exmatrix = exmatrix
        self._resmatrix = resmatrix
        self._effectmatrix = effectmatrix

    @property
    def exmatrix(self):
        return self._exmatrix

    @property
    def resmatrix(self):
        return self._resmatrix

    @property
    def analysis_result(self):
        return METNamedTuple(effectmatrix=self._effectmatrix)
