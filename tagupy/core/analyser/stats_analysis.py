"""
Super Class of any Statistical Analysis Module
"""

import numpy as np

# from tagupy.experiment import MatrixGenerator


class StatsAnalysis():
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

    # @classmethod
    # def from_matrix_generator(
    #     matrix_generator: MatrixGenerator,
    #     result: np.ndarray
    # ):
    #     """
    #     Recieve the workflow data from experiment planning part
    #
    #     Parameters
    #     ----------
    #     matrix_generator: MatrixGenerator
    #         Planning buffer data
    #
    #     result: np.ndarray
    #         Result data bound for experiment matrix in `matrix_generator`
    #     """
    #     pass
