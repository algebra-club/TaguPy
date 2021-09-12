'''
_Analyser Class of Linear Regression Analysis
'''

import numpy as np
import statsmodels.regression.linear_model as smlm
import statsmodels.api as sm
from tagupy.type import _Analyzer as Analyzer
from typing import Callable, NamedTuple


class LAResult(NamedTuple):
    '''
    NamedTuple Class of Linear Regression Analysis Results
    '''
    exmatrix: np.ndarray
    resmatrix: np.ndarray
    model: smlm.RegressionResultsWrapper
    params: np.ndarray
    bse: np.ndarray
    predict: Callable
    rsquared: float
    summary: Callable


class LinearAnalysis(Analyzer):
    '''
    Analyzer Module of Linear Analysis

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
            | Ordinary Least Square | OLS |
            | Weighted Least Square | WLS|
            | Generalized Least Square | GLS |

        """
        assert model in ["OLS", "WLS", "GLS"], \
            f"model expected 'OLS', 'WLS', or 'GLS', got {model}"
        self.model = {"OLS": (0, sm.OLS), "WLS": (1, sm.WLS), "GLS": (2, sm.GLS)}[model]

    def analyze(
        self,
        exmatrix: np.ndarray,
        result: np.ndarray,
        add_const: bool = True,
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
        if not isinstance(exmatrix, np.ndarray):
            raise TypeError(f"exmatrix expected numpy.ndarray; got {type(exmatrix)}")
        if not isinstance(result, np.ndarray):
            raise TypeError(f"result expected numpy.ndarray; got {type(result)}")
        assert exmatrix.shape[0] == result.shape[0], \
            f"numbers of rows in exmatrix & result should be equal; \
                got {exmatrix.shape[0]} & {result.shape[0]}"
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
        ret_model = self.model[1](**kwargs)
        res = ret_model.fit()
        return LAResult(
            exmatrix=exmatrix,
            resmatrix=result,
            model=ret_model,
            params=res.params,
            bse=res.bse,
            predict=res.predict,
            rsquared=res.rsquared,
            summary=res.summary
        )
