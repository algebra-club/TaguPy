'''
_Generator Class of Definitive Screening Design Generator Module
'''
import numpy as np

from tagupy.design.generator import _dsd_ref as ref
from tagupy.type import _Generator as Generator
from tagupy.utils import is_positive_int


class DSD(Generator):
    '''
    Generator Class of Definitive Screening Design Generator Module

    Method
    ------
    get_exmatrix(self, n_factor: int, n_fake: int) -> np.ndarray

    Note
    ----
    Definitive screening designs (DSDs) are three-level designs
    for studying m quantitative factors with the following main desirable properties:
    1. The design is mean orthogonal.
    2. The number of required runs is n = 2m + 1 (2m + 3 when m is odd), that is,
    saturated for estimating the intercept, m main effects, and m quadratic effects.
    3. Unlike resolution III designs, all main effects are orthogonal to all two-factor
    interactions.
    4. Unlike resolution IV designs, two-factor interactions are not fully aliased with one
    another.

    see also:
    Jones, B., & Nachtsheim, C. J. (2017).
    A Class of Three-Level Designs for Definitive Screening in the Presence of Second-Order Effects
    A Quarterly Journal of Methods, Applications and Related Topics Volume 43, 2011
    https://doi.org/10.1080/00224065.2011.11917841
    '''

    def __init__(self, n_rep: int):
        '''
        Parameters
        ----------
        n_rep: int
            number of replications; that value is applied
            when the whole set of experiment is replicated
            for the quality assurance of the experiment data.
            (when n_rep = 1, it implies that a single run
            for each condition will be planed)
        '''
        assert is_positive_int(n_rep),\
            f"Invalid input: n_rep expected positive (>0) integer, got {type(n_rep)}::{n_rep}"
        self.n_rep = n_rep

    def get_exmatrix(self, n_factor: int, n_fake: int) -> np.ndarray:
        '''
        create a definitive screening design

        Parameters
        ----------
        n_factor: int
            number of factors used in the experiment
        n_fake: int
            number of fake factors which enable better estimation

        Returns
        -------
        ex_mat(: np.ndarray())
            experiment matrix
            if n_factor+n_fake is even, shape of ex_mat would be (2*(n_factor+n_fake)+1, n_factor)
            if n_factor+n_fake is odd, shape of ex_mat would be (2*(n_factor+n_fake)+3, n_factor)

        Note
        ----
        3 <= n_factor + n_fake <= 50 is expected
        as for the number of fake factor, n_fake = 2 is used as standard

        Example
        -------
        >>> from tagupy.design.generator import DSD
        >>> model = DSD(n_rep=2)
        >>> model.get_exmatrix(n_factor=6, n_fake=2)
        array([[ 0, -1, -1, -1, -1, -1],
               [ 1,  0,  1,  1, -1,  1],
               [ 1, -1,  0,  1,  1, -1],
               [ 1, -1, -1,  0,  1,  1],
               [ 1,  1, -1, -1,  0,  1],
               [ 1, -1,  1, -1, -1,  0],
               [ 1,  1, -1,  1, -1, -1],
               [ 1,  1,  1, -1,  1, -1],
               [ 0,  1,  1,  1,  1,  1],
               [-1,  0, -1, -1,  1, -1],
               [-1,  1,  0, -1, -1,  1],
               [-1,  1,  1,  0, -1, -1],
               [-1, -1,  1,  1,  0, -1],
               [-1,  1, -1,  1,  1,  0],
               [-1, -1,  1, -1,  1,  1],
               [-1, -1, -1,  1, -1,  1],
               [ 0,  0,  0,  0,  0,  0],
               [ 0, -1, -1, -1, -1, -1],
               [ 1,  0,  1,  1, -1,  1],
               [ 1, -1,  0,  1,  1, -1],
               [ 1, -1, -1,  0,  1,  1],
               [ 1,  1, -1, -1,  0,  1],
               [ 1, -1,  1, -1, -1,  0],
               [ 1,  1, -1,  1, -1, -1],
               [ 1,  1,  1, -1,  1, -1],
               [ 0,  1,  1,  1,  1,  1],
               [-1,  0, -1, -1,  1, -1],
               [-1,  1,  0, -1, -1,  1],
               [-1,  1,  1,  0, -1, -1],
               [-1, -1,  1,  1,  0, -1],
               [-1,  1, -1,  1,  1,  0],
               [-1, -1,  1, -1,  1,  1],
               [-1, -1, -1,  1, -1,  1],
               [ 0,  0,  0,  0,  0,  0]])
        '''
        assert is_positive_int(n_factor),\
            f"Invalid input: n_factor expected positive (>0) integer,\
                 got {type(n_factor)}::{n_factor}"
        assert is_positive_int(n_fake),\
            f"Invalid input: n_fake expected positive (>0) integer,\
                 got {type(n_fake)}::{n_fake}"

        sum_fac = n_factor + n_fake
        assert 3 <= sum_fac <= 50,\
            f"Invalid input: sum of n_factor and n_fake expected 3 <= & <= 50, got {sum_fac}"
        if sum_fac % 2:
            sum_fac += 1
        l_func = [
            ((4, 6, 8, 12, 14, 18, 20, 24, 30, 32, 38, 42, 44, 48), ref._cmateq5),
            ((10, 22, 26, 34, 50), ref._cmateq2),
            ((16, 40), ref._dsddb),
            ((28, 36), ref._dsdeq3),
            ((46, 46), ref._dsd46),
            ]

        for num, func in l_func:
            if sum_fac in num:
                if sum_fac != 46:
                    c_mat = func(sum_fac, ref._gen_vec)
                else:
                    c_mat = func()
        ex_mat = ref._get_dsd(n_factor=n_factor, c_mat=c_mat)

        return np.vstack([ex_mat] * self.n_rep)
