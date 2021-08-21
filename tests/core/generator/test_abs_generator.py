"""Test of _Generator, super class of every matrix generator module"""

import numpy as np
import pytest

from tagupy.core import _Generator


def test_constructor():
    arg = {
        "n_factor": 3,
        "n_level": 5,
        "mode": "cont"
    }

    temp = _Generator(**arg)
    assert hasattr(temp, "exmatrix"), "exmatrix must be set"


def test_constructor_level():
    arg = {
        "n_factor": 3,
        "n_level": 1,
        "mode": "cont"
    }

    with pytest.raises(AssertionError):
        _Generator(**arg)


def test_constructor_mode():
    args = [
        {
            "n_factor": 3,
            "n_level": 5,
            "mode": "cont"
        },
        {
            "n_factor": 3,
            "n_level": 5,
            "mode": "cat"
        },
        {
            "n_factor": 3,
            "n_level": 5,
            "mode": ""
        },
        {
            "n_factor": 3,
            "n_level": 5,
            "mode": "nanachi"
        }
    ]

    for arg in args[:-1]:
        temp = _Generator(**arg)
        assert hasattr(temp, "exmatrix"), "exmatrix must be set"

    with pytest.raises(AssertionError):
        _Generator(**(args[3]))


def test_load_dict():
    arg = {
        'nanachi': (3, 'cont'),
        'bananachi': (2, 'cont'),
        'banananachi': (5, 'cont'),
    }
    temp = _Generator.load_dict(arg)

    assert isinstance(temp, _Generator)


# def test_load_dict_level():
#     arg = {
#         'nanachi': (3, 'cont'),
#         'bananachi': (1, 'cont'),
#         'banananachi': (5, 'cont'),
#     }
#     with pytest.raises(AssertionError):
#         _Generator.load_dict(arg)
#
#
# def test_load_dict_mode():
#     arg = {
#         'nanachi': (3, 'cont'),
#         'bananachi': (2, 'cont'),
#         'banananachi': (5, 'honya'),
#     }
#     with pytest.raises(AssertionError):
#         _Generator.load_dict(arg)


def test_get_alias_matrix():
    arg = {
        "n_factor": 3,
        "n_level": 5,
        "mode": "cont"
    }

    temp = _Generator(**arg)
    assert isinstance(temp.get_alias_matrix(10), np.ndarray)
