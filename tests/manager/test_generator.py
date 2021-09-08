from tagupy.manager.generator import (
        OneHot,
        FullFact,
        WrapperGenerator,
)

import pytest
import numpy as np
import click

import inspect

from typing import Type, Dict, Any, List


def _test_generate(cls: Type[WrapperGenerator], args: List[Dict[str, Any]]):
    assert issubclass(cls, WrapperGenerator), \
        f'{cls} must be a subclass of WrapperGenerator'

    for a in args:
        try:
            res = cls.generate(**a)
            assert isinstance(res, np.ndarray)
        except Exception as e:
            pytest.fail(f'Unexpected error occured at {a}, got msg below\n{e.args}')

    info = inspect.getfullargspec(cls.generate)
    res = cls.required_params()

    try:
        assert len(info.args) - 1 == len(res.keys())
        for argname, clicktype in res.values():
            assert argname in info.args
            assert isinstance(clicktype, click.types.ParamType)
    except Exception as e:
        pytest.fail(f'{e.args}\nNote: Argname may not be matched({info.args, res.keys()})')


def test_OneHot():
    args = [
        {'n_rep': 3, 'n_factor': 5},
        {'n_rep': 1, 'n_factor': 20},
        {'n_rep': 5, 'n_factor': 10},
    ]
    _test_generate(OneHot, args)


def test_FullFact_inheritance():
    args = [
        {'n_rep': 3, 'levels': [2, 3, 2]},
        {'n_rep': 1, 'levels': [3, 3, 3, 3]},
        {'n_rep': 5, 'levels': [2, 2, 2, 2, 2]},
    ]
    _test_generate(FullFact, args)
