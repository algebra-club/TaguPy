"""
_Generator Class of One Hot Design Generator Module
"""

import numpy as np

from typing import Any, Dict
from .abs_generator import _Generator


class OneHot(_Generator):
    """
    _Generator Class of One Hot Design Generator Module

    Method
    ------
    get_exmatrix(**info: Dict[str, Any]) -> np.ndarray
    get_alias_matrix(max_dim: int) -> np.ndarray
    """
    n_rep: int = 1
    n_factor: int = 1
    exmatrix: np.ndarray = np.array([])

    def __init__(self, **kwargs: Dict[str, Any]):
        """
        Parameters
        ----------
        kwargs: Dict[str, Any]
            it is expected to contain the following info

            1. n_rep: int
                number of replications; that value is applied
                when the whole set of experiment is replicated
                for the sake of quality assurement of the experiment data.
                default value is 1
        """
        assert list(kwargs.keys()) == ["n_rep"], \
            f"Erorr: unexpected keys {kwargs.keys()}, expected ['n_rep']"
        assert type(kwargs["n_rep"]) == int, \
            "Error: n_rep expected int"
        assert kwargs["n_rep"] >= 1, \
            f"Error: n_rep expected non-negative int, got {kwargs['n_rep']}"
        self.n_rep = kwargs["n_rep"]
        return None

    def get_exmatrix(self, **info: Dict[str, Any]) -> np.ndarray:
        """
        Generate One Hot Design Matrix

        Parameters
        ----------
        info: Dict[str, Any]
            it is expected to have the following info

            1. n_factor: int
                number of factors you use in this experiment

        Return
        ------
        exmatrix: np.ndarray
            Experiment Matrix (n_experiment x n_factor)

        Notes
        -----
        Suppose you have 5 factors with 2 levels (+/-) each
        to consider in the experiment and you are interested
        in the single effects on the outcome.

        You may consider to try put in (or turn on,
        depends on the experiment protocols)
        one-by-one and the rest factors would be
        removed from the conditions.

        Here we defined this way of experiment design as
        One Hot Design Matrix, as each row of the it
        would be an one hot vector.

        To assure the reliability of experiment, we
        reccomend you to replicate the same conditions
        and acquire multiple sets of the data.

        You can have the replicated One Hot Design Matrix
        at one time by setting n_rep as large
        non-negative integer as you like.

        """
        assert list(info.keys()) == ["n_factor"], \
            f"Erorr: unexpected keys {info.keys()}, expected ['n_factor']"
        assert type(info["n_factor"]) == int, \
            "Error: n_factor expected int"
        assert info["n_factor"] >= 1, \
            f"Error: n_factor expected non-negative int"
        self.n_factor = info["n_factor"]
        _res = np.concatenate(
            [
                np.identity(
                    self.n_factor, dtype=int
                    ) for i in range(self.n_rep)
                    ]
            )
        self.exmatrix = _res
        return self.exmatrix

    def get_alias_matrix(self, max_dim: int) -> np.ndarray:
        """
        Return Alias Matrix

        Parameters
        ----------
        max_dim: int
            maximum dimension treated in alias matrix

        Return
        ------
        alias matrix: numpy.ndarray
            Alias Matrix (n_factor x n_factor)

        Notes
        -----
        https://community.jmp.com/t5/JMP-Blog/What-is-an-Alias-Matrix/ba-p/30448
        """
