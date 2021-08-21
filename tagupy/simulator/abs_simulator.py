"""
ABC Class of Simulator
This package enable the users to check the charactors of each DoE methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

__all__ = [
    "_Simulator"
]


class _Simulator(ABC):
    """
    ABC Class of DoE Simulator
    """

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Parameters
        ----------
        kwargs: Dict[str, Any]
            Parameters of each simulator
        """
        pass

    @abstractmethod
    def simulate(self, exmatrix: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        exmatrix: np.ndarray
            Target experiment matrix (n_experiment x n_factor)

        Returns
        -------
        result_matrix: np.ndarray
            Simulation result (n_experiment x 1)
        """
        pass
