"""
_Generator Class of Plackett-Burman Design Generator Module
"""

import numpy as np

from tagupy.type import _Generator as Generator
from tagupy.utils import is_positive_int
from tagupy.design.generator import _pb_ref as ref


class PlackettBurman(Generator):
    """
    Generator Class of Plakett-Burman Design Generator Module

    Method
    ------
    get_exmatrix(**info: Dict[str, Any]) -> np.ndarray

    Notes
    -----
    get_exmatrix function returns experiment matrix of Plackett-Burman design
    Although Plackett-Burman (PB) design requires single runs for all conditions,
    this module supports replicated design for experiments deal with instable results.
    PB design supports experiments with 4 to 99 factors expect 92.

    see also:
    https://en.wikipedia.org/wiki/Plackett%E2%80%93Burman_design

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
        Generate Plakett-Burman Design Matrix

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
        >>> from tagupy.design import PlackettBurman
        >>> model = PlackettBurman(n_rep=2)
        >>> model.get_exmatrix(n_factor=3)
        array([[ 1,  1, -1],
               [-1,  1,  1],
               [ 1, -1,  1],
               [-1, -1, -1],
               [ 1,  1, -1],
               [-1,  1,  1],
               [ 1, -1,  1],
               [-1, -1, -1]])
        """
        assert is_positive_int(n_factor), \
            f"Invalid input: n_factor expected positive (>0) int, got {type(n_factor)}::{n_factor}"
        assert n_factor < 100, \
            f"Invalid input: n_factor is supported for int < 100, got {n_factor}"

        l_func = [[
            ref._pb4, ref._pb8, ref._pb12, ref._pb16, ref._pb20,
            ref._pb24, ref._pb28, ref._pb32, ref._pb36, ref._pb40,
            ref._pb44, ref._pb48, ref._pb52, ref._pb56, ref._pb60,
            ref._pb64, ref._pb68, ref._pb72, ref._pb76, ref._pb80,
            ref._pb84, ref._pb88, ref._pb92, ref._pb96, ref._pb100
        ][i//4] for i in range(100)]
        res = np.vstack([l_func[n_factor]()[:, :n_factor]] * self.n_rep)
        return res
