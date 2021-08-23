"""
_Generator Class of One Hot Design Generator Module
"""

import numpy as np

from .abs_generator import _Generator


class OneHot(_Generator):
    """
    Generator Class of One Hot Design Generator Module

    Method
    ------
    get_exmatrix(**info: Dict[str, Any]) -> np.ndarray
    get_alias_matrix(max_dim: int) -> np.ndarray

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
    One Hot Design Matrix, as each row of the it
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
        self.n_rep = n_rep \
            if isinstance(n_rep, int) else int(n_rep)
        assert self.n_rep >= 1, \
            "Error: n_rep expected integer >= 1"

    def get_exmatrix(self, n_factor: int) -> np.ndarray:
        """
        Generate One Hot Design Matrix

        Parameters
        ----------
        n_factor: int
            number of factors you use in this experiment
            As this method is limited for multifactorial experiment,
            n_factor expects integer >= 2

        Return
        ------
        exmatrix: np.ndarray
            Experiment Matrix (n_experiment x n_factor)
        """
        f = n_factor \
            if isinstance(n_factor, int) else int(n_factor)
        assert f >= 2, \
            "Error: n_factor expected integer >= 2"
        _res = np.concatenate(
            [np.identity(
                f,
                dtype=int
                ) for i in range(self.n_rep)]
        )
        self.exmatrix = _res
        return _res

    def get_alias_matrix(self) -> np.ndarray:
        """
        Return Alias Matrix

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
        return self.exmatrix \
            # alias matrixの実装について相談したいので、一旦保留にします
