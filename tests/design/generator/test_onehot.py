"""
Test for One Hot Design Generator Module
"""

import numpy as np
import pytest

from tagupy.core.generator.onehot import OneHot


def test_init_input_dtype():
    arg = "moge"
    with pytest.raises(AssertionError):
        OneHot(arg)


def test_init_input_value():
    arg = 0
    with pytest.raises(AssertionError):
        OneHot(arg)


def test_init_input_key():
    kwargs = {
        "n_rep": 3,
        "moge": 46
    }
    with pytest.raises(TypeError):
        OneHot(**kwargs)


def test_init_default_value():
    assert OneHot().n_rep == 1, \
        "Error: default value for n_rep is not 1"


def test_exmatrix_input_dtype():
    n_rep = 1
    arg = "moge"
    with pytest.raises(AssertionError):
        OneHot(n_rep).get_exmatrix(arg)


def test_exmatrix_input_value():
    n_rep = 1
    arg = 0
    with pytest.raises(AssertionError):
        OneHot(n_rep).get_exmatrix(arg)


def test_exmatrix_input_key():
    n_rep = 1
    info = {
        "n_factor": 3,
        "moge": 46
    }
    with pytest.raises(TypeError):
        OneHot(n_rep).get_exmatrix(**info)


def test_exmatrix_null_input():
    n_rep = 1
    arg = {}
    with pytest.raises(AssertionError):
        OneHot(n_rep).get_exmatrix(arg)


def test_exmatrix_dtype():
    n_rep = 1
    arg = 2
    assert type(OneHot(n_rep).get_exmatrix(arg)) == np.ndarray, \
        "Error: dtype of exmatrix is not np.ndarray"


def test_exmatrix_shape():
    n_rep = 12
    n_factor = 28
    _model = OneHot(n_rep).get_exmatrix(n_factor)
    assert _model.shape == (n_factor*n_rep, n_factor), \
        "Error: something is wrong with exmatrix shape"


def test_exmatrix_identity():
    n_rep = 1
    n_factor = 2
    _model = OneHot(n_rep).get_exmatrix(n_factor)
    exp = np.array([[12, 45], [-6, 0]])
    assert ((_model @ exp) == exp).all(), \
        "Error: exmatrix should be identity matrix when n_rep == 1"


def test_exmatrix_tandem():
    n_rep = 2
    n_factor = 5
    _model = OneHot(n_rep).get_exmatrix(n_factor)
    _model_sep = [
        _model[:n_factor][:],
        _model[n_factor:][:]
        ]
    assert (_model_sep[0] == _model_sep[1]).all(), \
        "Error: something is wrong with replication"


def test_alias_input_dtype():
    n_rep = 1
    n_factor = 2
    arg = "moge"
    _model = OneHot(n_rep)
    _model.get_exmatrix(n_factor)
    with pytest.raises(AssertionError):
        _model.get_alias_matrix(arg)


def test_alias_input_min_val():
    n_rep = 1
    n_factor = 2
    arg = 0
    _model = OneHot(n_rep)
    _model.get_exmatrix(n_factor)
    with pytest.raises(AssertionError):
        _model.get_alias_matrix(arg)


def test_alias_input_max_val():
    n_rep = 3
    n_factor = 14
    arg = 2
    _model = OneHot(n_rep)
    _model.get_exmatrix(n_factor)
    with pytest.raises(AssertionError):
        _model.get_alias_matrix(arg)


def test_alias_dtype():
    n_rep = 1
    n_factor = 5
    dim = 1
    _model = OneHot(n_rep)
    _model.get_exmatrix(n_factor)
    assert type(_model.get_alias_matrix(dim)) == np.ndarray, \
        "Error: dtype of alias matrix is not np.ndarray"


def test_alias_matrix():
    n_rep = 1
    n_factor = 3
    arg = 1
    _model = OneHot(n_rep)
    _model.get_exmatrix(n_factor)
    exp = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    _res = _model.get_alias_matrix(arg)
    assert (exp == _res).all(), \
        "Error: something is wrong with alias_matrix"


def test_alias_default_value():
    n_rep = 5
    n_factor = 2
    _model = OneHot(n_rep)
    _model.get_exmatrix(n_factor)
    exp = _model.get_alias_matrix(max_dim=1)
    _res = _model.get_alias_matrix()
    assert (_res == exp).all(), \
        "Error: something is wrong with default value for max_dim"
