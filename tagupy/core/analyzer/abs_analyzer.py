"""
Super Class of any Statistical Analysis Module
"""

__all__ = [
    '_Analyzer',
]

import numpy as np


class _Analyzer():
    """
    Super Class of any Statistical Analysis codes

    Attributes
    ----------
    exmatrix: numpy.ndarray
        Target experiment Matrix
    result: numpy.ndarray
        Result Matrix related to experiment matrix
    """

    exmatrix: np.ndarray = None
    result: np.ndarray = None

    def __init__(self, exmatrix: np.ndarray, result: np.ndarray):
        """
        Parameters
        ----------
        exmatrix: numpy.ndarray
            Target experiment Matrix

        result: numpy.ndarray
            Target Result Matrix
        """
        msg = "Argument matrix must be the same size, got "
        assert exmatrix.shape == result.shape, \
            msg + f'{exmatrix.shape}, {result.shape}'
        pass
