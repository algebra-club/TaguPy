import pytest
import numpy as np
import pandas as pd
from tagupy.design.analyzer import EDA
from tagupy.design.analyzer._eda import EDAResult


@pytest.fixture
def valid_input():
    return np.array([
        [1, 2, 3],
        [4, 5, 6],
    ])


def test_init_valid_input(valid_input):
    analyzer = EDA(valid_input)

    assert isinstance(analyzer.dataframe, pd.DataFrame), \
        f'analyzer.dataframe expected pandas.DataFrame, \
            got {type(analyzer.dataframe)::{analyzer.dataframe}}'


def test_init_invlaid_dtype():
    args = [1, 'string', ('tuple'), [1, 2, 3]]

    for arg in args:
        with pytest.raises(TypeError):
            EDA(arg)


def test_analyze_valid_return(valid_input):
    analyzer = EDA(valid_input)

    result = analyzer.analyze()

    assert isinstance(result, EDAResult), \
        f'EDA.analyze should return EDAResult, {type(result)}::{result}'
