"""
ABC Class of any Statistical Analysis Module
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

import numpy as np

from .analyze_result import AnalyzeResult

__all__ = [
    '_Analyzer',
]


class _Analyzer(ABC):
    """
    ABC Class of any Statistical Analysis codes

    Methods
    -------
    analyze(exmatrix: np.ndarray, result: np.ndarray) -> AnalyzeResult
    """

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Parameters
        ----------
        kwargs: Dict[str, Any]
            Analysis params for each methods
        """
        pass

    @abstractmethod
    def analyze(
        self,
        exmatrix: np.ndarray,
        result: np.ndarray
    ) -> AnalyzeResult:
        """
        Analyze data from experiment matrix and result matrix

        Parameters
        ----------
        exmatrix: numpy.ndarray
            Target experiment Matrix (n_experiment x n_factor)

        result: numpy.ndarray
            Target Result Matrix (n_experiment x 1)

        Returns
        -------
        analysis_result: AnalysisResult
            Resulf of analysis method
        """
        pass
