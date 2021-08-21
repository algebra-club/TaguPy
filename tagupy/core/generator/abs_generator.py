"""
Super Class of any Experiment Matrix Generating Module
"""

from typing import Dict, Tuple

import numpy as np

__all__ = [
    "_Generator"
]


class _Generator():
    """
    Super Class of Experiment Matrix Generator

    Attributes
    ----------
    exmatrix: np.ndarray
        Experiment Matrix (n_experiment x n_factor)

    ClassMethod
    -----------
    load_dict(content: Dict[str, Tuple[int, str]])
    from_stats_analysis(cls, analysis: StatsAnalysis):

    Method
    ------
    get_alias_matrix(max_dim: int) -> np.ndarray
    """

    exmatrix: np.ndarray = None

    def __init__(self, n_factor: int, n_level: int, mode: str = ""):
        """
        Parameters
        ----------
        n_factor: int
            number of factors you use in this experiment

        n_level: int
            number of level, requires every factor has same level.

        mode: str default = "mode"
            mode of factor, 'cont' or 'cat.' or ''
            requires every factor is under the same mode.


        Notes
        -----
        If you use odd mode or level experiments, use load_dict() instead.
        """

        assert mode in ('cont', 'cat', ''), \
            f'Mode should be "cont" or "cat", got {mode}'

        assert n_level != 1, 'Are you Sure???? one level???'

    @classmethod
    def load_dict(cls, content: Dict[str, Tuple[int, str]]):
        """
        Construct MatrixGenerator with custom complex dictionary object.

        Parameters
        ----------
        content: Dict[str, Tuple[int, str]]
            complex experiment target
            ex. one categorical factor and 2 continual factors
        """

        return cls(0, 0)

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
            Alias Matrix

        Notes
        -----
        https://community.jmp.com/t5/JMP-Blog/What-is-an-Alias-Matrix/ba-p/30448
        """
        return np.array([])
