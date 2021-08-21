"""
ABC Class of any Experiment Matrix Generating Module
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np

__all__ = [
    "_Generator"
]


class _Generator(ABC):
    """
    ABC Class of Experiment Matrix Generator

    Method
    ------
    get_exmatrix(**info: Dict[str, Any]) -> np.ndarray
    get_alias_matrix(max_dim: int) -> np.ndarray
    """

    @abstractmethod
    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Parameters
        ----------
        kwargs: Dict[str, Any]
            generate method options of each Generator
        """
        pass

    @abstractmethod
    def get_exmatrix(self, **info: Dict[str, Any]) -> np.ndarray:
        """
        Generate Experiment Matrix

        Parameters
        ----------
        info: Dict[str, Any]
            it is expected to contain following info

            1. n_factor: int
                number of factors you use in this experiment
            2. n_level: int
                number of levels, requires every factor has the same level.
            3. mode: str default = "mode"
                mode of factor, 'cont' or 'cat' or ''
                requires every factor is under the same mode.

        Return
        ------
        exmatrix: np.ndarray
            Experiment Matrix (n_experiment x n_factor)
        """
        pass

    @abstractmethod
    def get_alias_matrix(self, max_dim: int) -> np.ndarray:
        """
        Return Alias Matrix

        Parameters
        ----------
        max_dim: int
            maximum dimension treated in alias matrix

        Returns
        -------
        alias matrix: numpy.ndarray
            Alias Matrix (n_factor x n_factor)

        Notes
        -----
        https://community.jmp.com/t5/JMP-Blog/What-is-an-Alias-Matrix/ba-p/30448
        """
        pass
