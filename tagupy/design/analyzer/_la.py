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
    resid: np.ndarray
    bse: np.ndarray
    predict: Callable
    rsquared: float
    summary: Callable


class LinearAnalysis(Analyzer):
    '''
    Analyzer Module of Linear Analysis

    Method
    ------
    analyze(exmatrix: np.ndarray, result:np.ndarray) -> NamedTuple

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
        exmatrix: numpy.ndarray
        result: numpy.ndarray
        add_const: bool, default False
        missing: str, default "none"
            parameter used in statsmodels.regression.linear_model
            Here is the statement cited from the official documment of
            statsmodels.regression.linear_model.OLS
            ```
            Available options are ‘none’, ‘drop’, and ‘raise’.
            If ‘none’, no nan checking is done.
            If ‘drop’, any observations with nans are dropped.
            If ‘raise’, an error is raised. Default is ‘none’.
            ```
        hasconst: None or bool, default None
            parameter used in statsmodels.regression.linear_model
            Here is the statement cited from the official documment of
            statsmodels.regression.linear_model.OLS
            ```
            Indicates whether the RHS includes a user-supplied constant.
            If True, a constant is not checked for and k_constant is set to 1
            and all result statistics are calculated as if a constant is present.
            If False, a constant is not checked for and k_constant is set to 0.
            ```
        weights: numpy.ndarray, default 1
            parameter used in statsmodels.regression.linear_model.WLS
            Unless you initialized LinearAnalysis class with LinearAnalysis(model="WLS"),
            whatever value you input will be ignored.
            Here is the statement cited from the official documment of
            statsmodels.regression.linear_model.WLS
            ```
            A 1d array of weights. If you supply 1/W then the variables are
            pre- multiplied by 1/sqrt(W). If no weights are supplied the default value is 1
            and WLS results are the same as OLS.
            ```
        sigma: float or numpy.ndarray, default None
            parameter used in statsmodels.regression.linear_model.GLS
            Unless you initialized LinearAnalysis class with LinearAnalysis(model="GLS"),
            whatever value you input will be ignored.
            Here is the statement cited from the official documment of
            statsmodels.regression.linear_model.GLS
            ```
            The array or scalar sigma is the weighting matrix of the covariance.
            The default is None for no scaling. If sigma is a scalar, it is assumed that
            sigma is an n x n diagonal matrix with the given scalar, sigma as the value of
            each diagonal element. If sigma is an n-length vector, then sigma is assumed
            to be a diagonal matrix with the given sigma on the diagonal.
            This should be the same as WLS.
            ```

        Notes
        -----
        see also:
        https://www.statsmodels.org/stable/api.html
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
            model=res,
            params=res.params,
            resid=res.resid,
            bse=res.bse,
            predict=res.predict,
            rsquared=res.rsquared,
            summary=res.summary
        )
