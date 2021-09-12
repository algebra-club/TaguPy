"""
Utility validators
"""
import numpy as np

from typing import Any, Iterable


__all__ = [
    "is_correct_id_list",
    "is_int_2d_array",
    "is_non_negative_int",
    "is_non_negative_int_list",
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
    return isinstance(arg, int) and arg > 0


def is_non_negative_int(arg: Any) -> bool:
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
    >>> from tagupy.utils import is_non_negative_int
    >>> is_non_negative_int(3)
    True
    >>> is_non_negative_int(0)
    True
    >>> is_non_negative_int(0.5)
    False
    >>> is_non_negative_int(-1)
    False
    """
    return isinstance(arg, int) and arg >= 0


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


def is_non_negative_int_list(arg: Any) -> bool:
    """
    Validate if the arg follows the correct form.

    Parameters
    ----------
    arg: Any
        arg is expected the following props.
        1. arg belongs to Iterable[int], being able to cast to List, and not empty
        2. Each element in arg, belongs to Integer and is not negative (0 >=).

    Retrun
    ------
    result: Bool
        Function returns True, if the arg following the expected props.
        Otherwise, it returns False.

    Examples
    --------
    >>> from tagupy.utils import is_non_negative_int_list
    >>> is_non_negative_int_list([1, 2, 3])
    True
    >>> is_non_negative_int_list([1, 2, 0])
    True
    >>> is_non_negative_int_list([1, 0.2, -3])
    False
    >>> is_non_negative_int_list('string')
    False
    >>> is_non_negative_int_list([])
    False
    """
    is_list = isinstance(arg, Iterable)
    length = sum(1 for _ in arg)
    is_empty = length > 0
    is_non_neg_int = sum(map(is_non_negative_int, arg)) == length

    return is_list and is_empty and is_non_neg_int


def is_correct_id_list(id: Any, matrix: np.ndarray) -> bool:
    '''
    Validate if the id follows the correct form

    Parameters
    ----------
    id: Any
        id is expected the following props.
        1. id belongs to Iterable[int], being able to cast to List, and not empty
        2. Each element in id, belongs to Integer and is not negative (0 >=).
        3. Each element in id does not overlap
        4. Each element in id　expected to be smaller than (column length -1)
    matrix: np.ndarray
        suppose exmatrix and resmatrix

    Return
    ------
    result: Bool
        Function returns True, if the arg following the expected props.
        Otherwise, it returns False.
    Examples
    --------
    >>> from tagupy.utils import is_correct_id_list
    >>> matrix = np.array([[1, 1, 0, 1],
    ...                    [1, 1, 1, 0],
    ...                    [1, 0, 1, 1],
    ...                    [1, 0, 0, 0]])
    >>> is_correct_id_list([0, 1, 2, 3], matrix)
    True
    >>> is_correct_id_list([0, 2], matrix)
    True
    '''
    is_nonneg_int_list = is_non_negative_int_list(id)
    assert is_nonneg_int_list, \
        f'id list expected list of non negative integer (0>=) got: {type(id)}:{id}'
    length = sum(1 for _ in id)
    is_non_dup = length == len(set(id))
    assert is_non_dup, \
        f'each element in id list expected not to be duplicated got: {id}'
    max_id = max(id)
    is_cor_id = max_id <= matrix.shape[1] - 1
    assert is_cor_id, \
        f'index in id list is out of range, expected <={matrix.shape[1] - 1} got:{max_id}'

    return is_nonneg_int_list and is_non_dup and is_cor_id


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
    >>> is_int_2d_array(np.array([[1, 2], [3, 4]]))
    True
    >>> is_int_2d_array([[1, 2], [3, 4]])
    False
    >>> is_int_2d_array([[1, 2], [3, 4.5]])
    False
    >>> is_int_2d_array([[]])
    False
    """

    if not isinstance(arg, np.ndarray):
        return False

    is_2d_array = arg.ndim == 2

    is_not_empty = arg.size != 0

    is_int_element = np.apply_along_axis(
        lambda row: [
            isinstance(element, (int, np.integer))
            for element in row
        ],
        axis=1,
        arr=arg,
    ).all()

    return is_2d_array and is_not_empty and is_int_element
