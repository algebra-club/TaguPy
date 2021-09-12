"""
Utility validators
"""
from typing import Any, Iterable


__all__ = [
    "is_positive_int",
    "is_positive_int_list",
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
    return (type(arg) == int) and arg > 0


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
    map(is_positive_int, arg)
    is_list = isinstance(arg, Iterable)
    length = sum(1 for _ in arg)
    is_empty = length > 0
    is_pos_int = sum(map(is_positive_int, arg)) == length

    return is_list and is_empty and is_pos_int
