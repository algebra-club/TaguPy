import pytest
import numpy as np
from tagupy.design.analyzer import MET
from tagupy.design.analyzer._met import METResult


@pytest.fixture
def valid_mock_input():
    return {
        'exmatrix': np.ones((4, 4), dtype=np.int64),
        'resmatrix': np.ones((4, 3)),
    }


@pytest.fixture
def invalid_exmatrix_value_input():
    return {
        'exmatrix': np.full((4, 4), 1.2),
        'resmatrix': np.ones((4, 3)),
    }


@pytest.fixture
def invalid_exmatrix_shape_input():
    return {
        'exmatrix': np.ones((1, 2), dtype=np.int64),
        'resmatrix': np.ones((3, 4)),
    }


def test_init_default_input():
    analysis = MET()

    assert analysis.max_dim == 1, \
        'default max_dim expected only integer value `1`'


def test_init_valid_input():
    analysis = MET(1)

    assert analysis.max_dim == 1, \
        'max_dim expected only integer value `1`'


def test_init_invalid_inputs():
    args = [2, 1.5, -1]

    for arg in args:
        with pytest.raises(AssertionError) as e:
            MET(arg)

    assert 'expected positive integer' in f'{e.value}', \
        f'Assertion message should contain reasons, got "{e.value}'


def test_analyze_invalid_input_exmatrix(invalid_exmatrix_value_input):
    analysis = MET()

    with pytest.raises(AssertionError) as e:
        analysis.analyze(**invalid_exmatrix_value_input)

    assert f'{invalid_exmatrix_value_input["exmatrix"]}' in f'{e.value}', \
        'Assertion message should contain reasons, got "{e.value}'


def test_analyze_invalid_input():
    analysis = MET()

    args = [
        {'exmatrix': 1, 'resmatrix': np.ones((3, 3))},
        {'exmatrix': np.ones((3, 3)), 'resmatrix': 1},
        {'exmatrix': [[1, 2], [3, 4]], 'resmatrix': [[5, 6], [7, 8]]},
    ]

    for arg in args:
        with pytest.raises(AssertionError) as e:
            analysis.analyze(**arg)

    assert 'matrix expected a numpy.ndarray' in f'{e.value}', \
        'Assertion message should contain reasons, got "{e.value}'


def test_analyze_valid_result_type(valid_mock_input):
    analysis = MET()
    result = analysis.analyze(**valid_mock_input)

    assert isinstance(result, METResult), \
        f'expected METResult, got {result}'


def test_analyze_valid_ressult_effectmatrix_shape(valid_mock_input):
    analysis = MET()
    result = analysis.analyze(**valid_mock_input)

    assert (4, 3) == result.effectmatrix.shape, \
        'got unexpected shape matrix from result.effectmatrix'

    assert (4, 4) == result.exmatrix.shape, \
        'got unexpected shape matrix from result.exmatrix'

    assert (4, 3) == result.resmatrix.shape, \
        'got unexpected shape matrix from result.resmatrix'


def test_analyze_invalid_shape():
    analysis = MET()

    with pytest.raises(ValueError):
        analysis.analyze(
            np.ones((2, 3), dtype=np.int64),
            np.ones((4, 5)),
        )
