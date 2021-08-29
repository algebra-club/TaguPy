"""
Data Structure for Analysis Results
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import numpy as np


class _AnalysisResult(ABC):
    """
    ABC Class of Statistical Analysis Result
    """
    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]):
        pass

    @property
    @abstractmethod
    def get_exmatrix(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def get_resmatrix(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def get_analysis_result(self) -> Dict[str, np.ndarray]:
        pass


AnalysisResult = Type[_AnalysisResult]
