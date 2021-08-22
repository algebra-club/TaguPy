"""
_Generator Class of One Hot Design Generator Module
"""

import numpy as np

from .abs_generator import _Generator


class OneHot(_Generator):
    """
    _Generator Class of One Hot Design Generator Module

    Method
    ------
    get_exmatrix(**info: Dict[str, Any]) -> np.ndarray
    get_alias_matrix(max_dim: int) -> np.ndarray
    """

    def __init__(self, n_rep: int = 1):
        """
        Parameters
        ----------
        n_rep: int
            number of replications; that value is applied
            when the whole set of experiment is replicated
            for the sake of quality assurement of the experiment data.
            default value is 1
        """
        self.n_rep = n_rep
        self.n_factor = None
        self.exmatrix = None
        assert type(self.n_rep) == int, \
            f"Error: n_rep expected int, got {type(self.n_rep)}"
        assert self.n_rep >= 1, \
            "Error: n_rep expected integer >= 1"
        return None

    def get_exmatrix(self, n_factor: int) -> np.ndarray:
        """
        Generate One Hot Design Matrix

        Parameters
        ----------
        n_factor: int
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
        self.n_factor = n_factor
        assert type(self.n_factor) == int, \
            f"Error: n_factor expected int, got {type(self.n_factor)}"
        assert self.n_factor >= 1, \
            "Error: n_factor expected integer >= 1"
        _res = np.concatenate(
            [
                np.identity(
                    self.n_factor, dtype=int
                    ) for i in range(self.n_rep)
                    ]
            )
        self.exmatrix = _res
        return self.exmatrix

    def get_alias_matrix(self, max_dim: int = 1) -> np.ndarray:
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

        Warning
        -------
        in One Hot Design, you can't calculate higher dimensional interactions,
        therefore max_dim expected to be 1
        """
        assert type(max_dim) == int, \
            f"Error: max_dim expected int, got {type(max_dim)}"
        assert max_dim == 1, \
            "Error: by the definition of One Hot Design, factor"
        assert max_dim <= self.n_factor, \
            "Error: value of max_dim should be lower than n_factor"
        return np.ones((self.n_factor, self.n_factor))
