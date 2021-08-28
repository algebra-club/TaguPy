import pytest
import numpy as np
from tagupy.design.analyzer import MainEffectTable


@pytest.fixture
def valid_input():
    return {
        'exmatrix': np.array(
            [[1, 1, 0, 1],
             [1, 1, 1, 0],
             [1, 0, 1, 1],
             [1, 0, 0, 0]]
        ),
        'result': np.array(
            [[3, 4, 7],
             [4, 9, 8],
             [5, 8, 3],
             [8, 3, 6]],
        )
    }


def test_init():
    MainEffectTable()


def test_analyze_valid_input(valid_input):
    analysis = MainEffectTable()
    result = analysis.analyze(**valid_input)

    exp = np.array([
        [0, 0, 0],
        [-1.5, 0.5, 1.5],
        [-0.5, 2.5, -0.5],
        [-1, 0, -1]
    ])

    np.testing.assert_array_equal(exp, result)
