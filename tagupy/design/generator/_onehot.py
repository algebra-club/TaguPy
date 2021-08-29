"""
_Generator Class of One Hot Design Generator Module
"""

import numpy as np

from tagupy.type import _Generator as Generator
from tagupy.utils import is_positive_int


class OneHot(Generator):
    """
    Generator Class of One Hot Design Generator Module

    Method
    ------
    get_exmatrix(**info: Dict[str, Any]) -> np.ndarray

    Notes
    -----
    Suppose you have multiple factors with 2 levels (+/-) each
    to consider in the experiment and you are interested
    in the single effects on the outcome.

    You may consider to try put in (or turn on,
    depends on the experiment protocols)
    one-by-one and the rest factors would be
    removed from the conditions.

    Here we defined this way of experiment design as
    One Hot Design Matrix, as each column of it
    would be an one hot vector.

    To assure the reliability of experiment, we
    reccomend you to replicate the same conditions
    and acquire multiple sets of the data.

    You can have the replicated One Hot Design Matrix
    at one time by setting n_rep as large
    non-zero natural number as you like.

    """

    def __init__(self, n_rep: int):
        """
        Parameters
        ----------
        n_rep: int
            number of replications; that value is applied
            when the whole set of experiment is replicated
            for the sake of quality assurance of the experiment data.
            (when n_rep = 1, it implies that a single run
            for each condition will be planed)
        """
        assert is_positive_int(n_rep), \
            f"Invalid input: n_rep expected positive (>0) integer, got {type(n_rep)}::{n_rep}"
        self.n_rep = n_rep

    def get_exmatrix(self, n_factor: int) -> np.ndarray:
        """
        Generate One Hot Design Matrix

        Parameters
        ----------
        n_factor: int
            number of factors you use in this experiment
            As this method is limited for multifactorial experiment,
            n_factor expects integer >= 1

        Return
        ------
        exmatrix: np.ndarray
            Experiment Matrix (n_experiment x n_factor)

        Example
        -------
        >>> from tagupy.design import OneHot
        >>> model = OneHot(n_rep=2)
        >>> model.get_exmatrix(n_factor=2)
        array([[1, 0],
               [0, 1],
               [0, 0],
               [1, 0],
               [0, 1],
               [0, 0]])
        """
        assert is_positive_int(n_factor), \
            f"Invalid input: n_factor expected positive (>0) int, got {type(n_factor)}::{n_factor}"

        res = np.vstack([np.identity(n_factor, int), np.zeros(n_factor, int)] * self.n_rep)
        return res
