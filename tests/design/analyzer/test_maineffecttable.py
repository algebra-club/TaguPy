import pytest
import numpy as np
from tagupy.design.analyzer import MainEffectTable


def test_init():
    MainEffectTable()


def test_analyze_invalid_input_exmatrix():
    analysis = MainEffectTable()
    arg = {
        'exmatrix': np.full((4, 4), -1),
        'resmatrix': np.ones((4, 3)),
    }

    with pytest.raises(AssertionError) as e:
        analysis.analyze(**arg)

    assert f'{arg["exmatrix"]}' in f'{e.value}', \
        'Assertion message should contain reasons, got "{e.value}'


def test_analyze_invalid_input():
    analysis = MainEffectTable()

    args = [
        {'exmatrix': 1, 'resmatrix': np.ones((3, 3))},
        {'exmatrix': np.ones((3, 3)), 'resmatrix': 1},
        {'exmatrix': [[1, 2], [3, 4]], 'resmatrix': [[5, 6], [7, 8]]},
    ]

    for arg in args:
        with pytest.raises(AssertionError) as e:
            analysis.analyze(**arg)

    assert 'matrix expected a np.ndarray' in f'{e.value}', \
        'Assertion message should contain reasons, got "{e.value}'


def test_analyze_valid_ressult_effectmatrix_shape():
    analysis = MainEffectTable()
    resmatrix = analysis.analyze(
        np.ones((4, 4)),
        np.ones((4, 3))
    )

    assert (4, 3) == resmatrix['effectmatrix'].shape, \
        f'got unexpected shape matrix from resmatrix.effectmatrix'


def test_analyze_invalid_shape():
    analysis = MainEffectTable()

    with pytest.raises(ValueError):
        analysis.analyze(
            np.ones((1, 2)),
            np.ones((3, 4)),
        )
