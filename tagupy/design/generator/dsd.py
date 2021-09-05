'''
_Generator Class of Definitive Screening Design Generator Module
'''
import numpy as np

from tagupy.type import _Generator as Generator
from tagupy.utils import is_positive_int, is_positive_int_list


class DSD(Generator):
    '''
    Generator Class of Definitive Screening Design Generator Module

    Method
    ------
    get_exmatrix(self, n_factors: int, n_fake: int) -> np.ndarray

    Note
    ----
    Definitive screening designs (DSDs) are three-level designs 
    for studying m quantitative factors with the following main desirable properties:
    1. The design is mean orthogonal.
    2. The number of required runs is n = 2m + 1 (2m + 3 when m is odd), that is, saturated for estimating the
    intercept, m main effects, and m quadratic effects. 
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
    
    def get_exmatrix(self, n_fac: int, n_fake: int) ->np.ndarray:
        '''
        c
        '''
    