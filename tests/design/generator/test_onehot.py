"""
Test for One Hot Design Generator Module
"""

import numpy as np
import pytest

from tagupy.core.generator.onehot import OneHot


def test_init_input_dtype():
    kwargs = {
        "n_rep": "moge"
    }
    with pytest.raises(AssertionError):
        OneHot(**kwargs)


def test_init_input_key():
    kwargs = {
        "n_rep": 3,
        "moge": 46
    }
    with pytest.raises(AssertionError):
        OneHot(**kwargs)


def test_init_input_value():
    kwargs = {
        "n_rep": 0
    }
    with pytest.raises(AssertionError):
        OneHot(**kwargs)


def test_exmatrix_input_dtype():
    kwargs = {
        "n_rep": 1
    }
    info = {
        "n_factor": "moge"
    }
    with pytest.raises(AssertionError):
        OneHot(**kwargs).get_exmatrix(**info)


def test_exmatrix_input_key():
    kwargs = {
        "n_rep": 1
    }
    info = {
        "n_factor": 356,
        "moge": 3
    }
    with pytest.raises(AssertionError):
        OneHot(**kwargs).get_exmatrix(**info)


def test_exmatrix_input_value():
    kwargs = {
        "n_rep": 1
    }
    info = {
        "n_factor": 0
    }
    with pytest.raises(AssertionError):
        OneHot(**kwargs).get_exmatrix(**info)


def test_exmatrix_dtype():
    kwargs = {
        "n_rep": 1
    }
    info = {
        "n_factor": 2
    }
    assert type(OneHot(**kwargs).get_exmatrix(**info)) == np.ndarray, \
        f"Error: dtype of exmatrix is not np.ndarray"


def test_exmatrix_shape():
    n_rep = 12
    n_factor = 28
    kwargs = {
        "n_rep": n_rep
    }
    info = {
        "n_factor": n_factor
    }
    (n_row, n_col) = (n_factor*n_rep, n_factor)
    _model = OneHot(**kwargs).get_exmatrix(**info)
    assert _model.shape == (n_row, n_col), \
        "Error: something is wrong with exmatrix shape"


def test_exmatrix_identity():
    n_rep = 1
    n_factor = 2
    kwargs = {
        "n_rep": n_rep
    }
    info = {
        "n_factor": n_factor
    }
    _model = OneHot(**kwargs).get_exmatrix(**info)
    exp = np.array([[12, 45], [-6, 0]])
    assert ((_model @ exp) == exp).all(), \
        "Error: exmatrix should be identity matrix when n_rep == 1"


def test_exmatrix_tandem():
    n_rep = 2
    n_factor = 5
    kwargs = {
        "n_rep": n_rep
    }
    info = {
        "n_factor": n_factor
    }
    _model = OneHot(**kwargs).get_exmatrix(**info)
    _model_sep = [
        _model[:n_factor][:],
        _model[n_factor:][:]
        ]
    assert (_model_sep[0] == _model_sep[1]).all(), \
        "Error: something is wrong with replication"
