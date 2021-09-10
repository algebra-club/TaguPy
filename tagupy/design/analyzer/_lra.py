'''
_Analyser Class of Linear Regression Analysis
'''

import numpy as np
import statsmodels
import statsmodels.api as sm
from tagupy.type import _Analyzer as Analyzer
from typing import Any, NamedTuple


class LinRegResult(NamedTuple):
    '''
    NamedTuple Class of Linear Regression Analysis Results
    '''
    exmatrix: np.ndarray
    resmatrix: np.ndarray
    model: statsmodels.regression.linear_model.RegressionResultsWrapper
    params: np.ndarray
    bse: np.ndarray
    predict: Any
    rsquared: float
    summary: Any


class LinReg(Analyzer):
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
    def __init__(self, model: str):
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

    def analyze(
        self,
        exmatrix: np.ndarray,
        result: np.ndarray,
        add_const: bool,
        missing: str = "none",
        hasconst: None or bool = None,
        weights: np.ndarray = 1,
        # defaul value for weights in sm.WLS() is 1 as well
        # default value will be converted into numpy.ndarray inside sm.WLS()
        sigma: float or np.ndarray = None
        # defaul value for sigma in sm.GLS() is None as well
        # both of float and default value will be converted into numpy.ndarray inside sm.GLS()
    ) -> NamedTuple:
        """
        Parameters
        ----------
        """
        kwargs = [
            {
                "endog": result,
                "exog": sm.add_constant(exmatrix) if add_const else exmatrix,
                "missing": missing,
                "hasconst": hasconst
            },
            {
                "endog": result,
                "exog": sm.add_constant(exmatrix) if add_const else exmatrix,
                "weights": weights,
                "missing": missing,
                "hasconst": hasconst
            },
            {
                "endog": result,
                "exog": sm.add_constant(exmatrix) if add_const else exmatrix,
                "sigma": sigma,
                "missing": missing,
                "hasconst": hasconst
            }
        ][self.model[0]]
        res = self.model[1](**kwargs)
        return LinRegResult(
            exmatrix=exmatrix,
            resmatrix=result,
            model=res,
            params=res.params,
            bse=res.bse,
            predict=res.predict,
            rsquared=res.rsquared,
            summary=res.summary
        )
