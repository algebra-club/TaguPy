'''
_Analyser Class of Linear Regression Analysis
'''

import numpy as np
import statsmodels
import statsmodels.api as sm
from tagupy.type import _Analyzer as Analyser
from typing import Any, NamedTuple


class LinRegResult(NamedTuple):
    exmatrix: np.ndarray
    resmatrix: np.ndarray
    model: statsmodels.regression.linear_model.RegressionResultsWrapper
    params: np.ndarray
    bse: np.ndarray
    predict: Any
    rsquared: float
    summary: Any


class LinReg(Analyser):
    '''
    _Analyser Class of Linear Regression Analysis

    Method
    ------
    analyze(exmatrix: np.ndarray, resylt:np.ndarray) -> AnalysisResult

    Notes
    -----
    see also:
    https://www.statsmodels.org/stable/examples/index.html
    '''
    def __init__(self, model:str):
        """
        Parameters
        ----------
        model: str
            model for Least Squares. This arg expects either "OLS", "WLS", or "GLS".
            see also the table below.

            | model | arguments |
            | :---: | :---:|
            | (ordinary) Least Square | OLS |
            | Weighted Least Square | WLS|
            | Generalized Least Square | GLS |

        """
        assert model in ["OLS", "WLS", "GLS"], \
            f"model expected 'OLS', 'WLS', or 'GLS', got {model}"
        self.model = {"OLS": (0, sm.OLS), "WLS": (1, sm.OLS), "GLS": (2, sm.GLS)}[model]
    
    # def analyze(self, exmatrix: np.ndarray, result: np.ndarray) -> NamedTuple:


    #     return LinRegResult()
