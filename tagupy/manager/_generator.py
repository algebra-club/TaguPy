from typing import List

import numpy as np

from tagupy.design.generator import (
    OneHot as _OneHot,
    FullFact as _FullFact,
)

from tagupy.type import PositiveInt, PositiveIntList


__all__ = [
    "OneHot",
    "FullFact",
]


class OneHot():
    """
    Wrapper object for cli
    """

    @classmethod
    def generate(cls, n_rep: int, n_factor: int) -> np.ndarray:
        return _OneHot(n_rep).get_exmatrix(n_factor)

    @classmethod
    def required_params(cls):
        return {
            "number of replication (ex. 2)": ("n_rep", PositiveInt),
            "number of factor (ex. 3)": ("n_factor", PositiveInt),
        }


class FullFact():
    """
    Wrapper object for cli
    """
    @classmethod
    def generate(cls, n_rep: int, levels: List[int]) -> np.ndarray:
        return _FullFact(n_rep).get_exmatrix(levels)

    @classmethod
    def required_params(cls):
        return {
            "number of replication (ex. 2)": ("n_rep", PositiveInt),
            "number of levels (ex. 1 2 3)": ("levels", PositiveIntList),
        }
