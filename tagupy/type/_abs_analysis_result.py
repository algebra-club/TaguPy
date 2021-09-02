"""
Abstract Data Structure for Analysis Results
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple

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
    def exmatrix(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def resmatrix(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def analysis_result(self) -> NamedTuple:
        pass
