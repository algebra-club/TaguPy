"""
Utility validators
"""
from typing import Any, Iterable
import numpy as np


__all__ = [
    "is_positive_int",
    "is_positive_int_list",
    "is_int_2d_array",
]


def is_positive_int(arg: Any) -> bool:
    """
    Validate if the arg follows the correct form.

    Parameters
    ----------
    arg: Any
        arg is expected the following props.
        1. Each element in arg, belongs to Integer, and is positive (>0).

    Retrun
    ------
    result: Bool
        Function returns True, if the arg following the expected props.
        Otherwise, it returns False.

    Examples
    --------
    >>> from tagupy.utils import is_positive_int
    >>> is_positive_int(3)
    True
    >>> is_positive_int(0.5)
    False
    >>> is_positive_int(0)
    False
    """
    return isinstance(arg, (int, np.integer)) and arg > 0


def is_positive_int_list(arg: Any) -> bool:
    """
    Validate if the arg follows the correct form.

    Parameters
    ----------
    arg: Any
        arg is expected the following props.
        1. arg belongs to Iterable[int], being able to cast to List, and not empty
        2. Each element in arg, belongs to Integer and is positive (>0).

    Retrun
    ------
    result: Bool
        Function returns True, if the arg following the expected props.
        Otherwise, it returns False.

    Examples
    --------
    >>> from tagupy.utils import is_positive_int_list
    >>> is_positive_int_list([1, 2, 3])
    True
    >>> is_positive_int_list([1, 0.2, 3])
    False
    >>> is_positive_int_list([1, 2, 0])
    False
    >>> is_positive_int_list('string')
    False
    >>> is_positive_int_list([])
    False
    """
    is_list = isinstance(arg, Iterable)
    length = sum(1 for _ in arg)
    is_empty = length > 0
    is_pos_int = sum(map(is_positive_int, arg)) == length

    return is_list and is_empty and is_pos_int


def is_int_2d_array(arg: Any):
    """
    Validate if the arg follows the correct form.

    Parameters
    ----------
    arg: Any
        arg is expected the following props.
        1. arg belongs to Iterable[int], being able to cast to List, and not empty
        2. Each element in arg, belongs to Integer

    Return
    ------
    result: Bool
        Function returns True, if the arg following the expected prpos.
        Othrwise, it returns False.


    Examples
    --------
    >>> from tagupy.utils import is_int_2d_array
    >>> is_int_2d_array([[1, 2], [3, 4]])
    True
    >>> is_int_2d_array([[1, 2], [3, 4.5]])
    False
    >>> is_int_2d_array(np.array([[1, 2], [3, 4]]))
    True
    >>> is_int_2d_array([[]])
    False

    """

    arg_ndarray = np.array(arg)

    is_2d_array = arg_ndarray.ndim == 2

    is_not_empty = arg_ndarray.size != 0

    is_int_element = np.apply_along_axis(
        lambda row: [
            isinstance(element, (int, np.integer))
            for element in row
        ],
        axis=1,
        arr=arg_ndarray,
    ).all()

    return is_2d_array and is_not_empty and is_int_element
