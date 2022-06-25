from abc import ABC, abstractclassmethod
from typing import Any, Dict, List, Tuple

import click
import numpy as np
from numpy.typing import ArrayLike

from tagupy.design.generator import (
    OneHot as _OneHot,
    FullFact as _FullFact,
)

from tagupy.type import PositiveInt, PositiveIntList


__all__ = [
    "OneHot",
    "FullFact",
]


class WrapperGenerator(ABC):
    def __init__(self):
        raise TypeError(f"Don't instantiate ABClass {type(self).__name__}")

    @classmethod
    @abstractclassmethod
    def generate(cls, **kwargs: Dict[str, Any]) -> ArrayLike:  # type: ignore
        pass  # pragma: no cover

    @classmethod
    @abstractclassmethod
    def required_params(cls) -> Dict[str, Tuple[str, click.ParamType]]:  # type: ignore
        pass  # pragma: no cover


class OneHot(WrapperGenerator):
    """
    Wrapper object for cli
    """

    @classmethod
    def generate(cls, n_rep: int, n_factor: int) -> np.ndarray:
        return _OneHot(n_rep).get_exmatrix(n_factor)

    @classmethod
    def required_params(cls) -> Dict[str, Tuple[str, click.ParamType]]:
        return {
            "number of replication (ex. 2)": ("n_rep", PositiveInt),
            "number of factor (ex. 3)": ("n_factor", PositiveInt),
        }


class FullFact(WrapperGenerator):
    """
    Wrapper object for cli
    """
    @classmethod
    def generate(cls, n_rep: int, levels: List[int]) -> np.ndarray:
        return _FullFact(n_rep).get_exmatrix(levels)

    @classmethod
    def required_params(cls) -> Dict[str, Tuple[str, click.ParamType]]:
        return {
            "number of replication (ex. 2)": ("n_rep", PositiveInt),
            "number of levels (ex. 1 2 3)": ("levels", PositiveIntList),
        }
