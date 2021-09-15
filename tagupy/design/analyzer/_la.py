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
        add_const: bool = False,
        missing: str = "none",
        hasconst: None or bool = None,
        weights: np.ndarray = 1,
        # defaul value for weights in sm.WLS() is 1 as well
        # default value will be converted into numpy.ndarray inside sm.WLS()
        sigma: float or np.ndarray = None,
        # defaul value for sigma in sm.GLS() is None as well
        # both of float and default value will be converted into numpy.ndarray inside sm.GLS()
        **kwargs
    ) -> NamedTuple:
        """
        Parameters
        ----------
        exmatrix: numpy.ndarray
            numpy.ndarray of experiment matrix.
            The shape of it is expected to be (rows, columns) = (number of runs, n_factor)
        result: numpy.ndarray
            numpy.ndarray of result matrix.
            The shape of it is expected to be (rows, columns) = (number of runs, 1)
        add_const: bool, default False
            if you would like to have constant term in the model, then input True,
            otherwise input False.
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
        **kwargs
            Additional keyword arguments for statsmodels.regression.linear_model
            Here is the statement cited from the official documment of
            statsmodels.regression.linear_model.OLS
            ```
            Extra arguments that are used to set model properties when using the formula interface.
            ```

        Return
        ------
        LAResult: NamedTuple
            results of statsmodels.regression.linear_model are packaged in a NamedTuple
            LAResult contains instances below:
            exmatrix: numpy.ndarray
                numpy.ndarray of experiment matrix.
                equivalent to "exog" of stasmodels.
            resmatrix: numpy.ndarray
                numpy.ndarray of result matrix.
                equivalent to "endog" of statsmodels.
            model: statsmodels.regression.linear_model.RegressionResultsWrapper
                model itself for the analysis.
                For further operation via certain methods of statsmodels,
                you can have access to the model here.
            params: numpy.ndarray
                coefficients of each parameter in the regression model.
                equivalent to either statsmodels.api.OLS(X, Y).fit().params,
                statsmodels.api.WLS(X, Y).fit().params, or statsmodels.api.GLS(X, Y).fit().params
            resid: numpy.ndarray
                numpy.ndarray of residal errors.
                equivalent to either statsmodels.api.OLS(X, Y).fit().resid,
                statsmodels.api.WLS(X, Y).fit().resid, or statsmodels.api.GLS(X, Y).fit().resid
            bse: numpy.ndarray
                numpy.ndarray of the standard erros of parameter estimates.
                equivalent to either statsmodels.api.OLS(X, Y).fit().bse,
                statsmodels.api.WLS(X, Y).fit().bse, or statsmodels.api.GLS(X, Y).fit().bse
            predict: Callable
                Calalble object of predicted values. You can get values by LAResult.predict().
                equivalent to either statsmodels.api.OLS(X, Y).fit().predict,
                statsmodels.api.WLS(X, Y).fit().predict, or statsmodels.api.GLS(X, Y).fit().predict
            rsquared: float
                R-squared value of the model.
                equivalent to either statsmodels.api.OLS(X, Y).fit().rsquared,
                statsmodels.api.WLS(X, Y).fit().rsquared,
                or statsmodels.api.GLS(X, Y).fit().rsquared
            summary: Callable
                Callable object of the regression result's summary.
                You can use it in the same way as
                statsmodels.regression.linear_model.OLSResults.summary.
                According to the official document of stasmodels,
                it has four parameters;
                ```
                yname: str, default None
                    Name of endogenous (response) variable. The default is 'y'.
                xname: list[str], default None
                    Names for the exogenous variables. Default is var_## for ##
                    in the number of regressors. Must match the number of parameters in the model.
                title: str, default None
                    Title for the top table. If not None, then this replaces the default title.
                alpha: float, default 0.05
                    The significance level for the confidence intervals.
                ```

        Notes
        -----
        see also:
        https://www.statsmodels.org/stable/api.html

        Examples
        --------
        >>> import numpy as np
        >>> from tagupy.design import LinearAnalysis
        >>> exmat = np.array([[0, 0, 1, 0, 1],
        ...                   [1, 0, 0, 0, 0],
        ...                   [0, 1, 0, 0, 1],
        ...                   [1, 1, 1, 0, 0],
        ...                   [0, 0, 1, 1, 0],
        ...                   [1, 0, 0, 1, 1],
        ...                   [0, 1, 0, 1, 0],
        ...                   [1, 1, 1, 1, 1]])
        >>> resmat = np.array([[2.9742447 ],
        ...                    [0.23557889],
        ...                    [1.67324722],
        ...                    [3.20927791],
        ...                    [1.98413631],
        ...                    [5.25113493],
        ...                    [1.75210261],
        ...                    [7.75030858]])
        >>> # Example for OLS
        >>> model1 = LinearAnalysis("OLS")
        >>> result1 = model1.analyze(exmatrix=exmat, result=resmat, add_const=True)
        >>> result1.params.round(2)
        array([-1.66,  2.02,  0.98,  1.75,  2.16,  2.62])
        >>> result1.resid.round(2)
        array([ 0.27, -0.12, -0.27,  0.12, -0.27,  0.12,  0.27, -0.12])
        >>> result1.predict().round(2)
        array([2.71, 0.35, 1.94, 3.09, 2.25, 5.13, 1.48, 7.87])
        >>> # Example for WLS
        >>> model2 = LinearAnalysis("WLS")
        >>> result2 = model2.analyze(exmatrix=exmat, result=resmat, weights=np.ones(len(resmat)))
        >>> result2.params.round(2)
        array([1.46, 0.43, 1.2 , 1.61, 2.06])
        >>> result2.resid.round(2)
        array([-0.29, -1.23, -0.82,  0.12, -0.82,  0.12, -0.29,  0.99])
        >>> result2.predict().round(2)
        array([3.26, 1.46, 2.49, 3.09, 2.81, 5.13, 2.04, 6.76])
        >>> # Example for GLS
        >>> model3 = LinearAnalysis("GLS")
        >>> sigma = np.array([[ 1.   , -0.119,  0.071,  0.002,  0.005,  0.   ,  0.   , -0.   ],
        ...                   [ 0.267,  1.   , -0.267,  0.014, -0.019,  0.   ,  0.001,  0.   ],
        ...                   [ 0.071, -0.119,  1.   ,  0.119,  0.071,  0.002,  0.005, -0.   ],
        ...                   [ 0.019,  0.014, -0.267,  1.   , -0.267,  0.014,  0.019,  0.   ],
        ...                   [ 0.005, -0.002,  0.071,  0.119,  1.   ,  0.119,  0.071, -0.002],
        ...                   [ 0.001,  0.   , -0.019,  0.014, -0.267,  1.   ,  0.267,  0.014],
        ...                   [ 0.   , -0.   ,  0.005,  0.002,  0.071,  0.119,  1.   , -0.119],
        ...                   [ 0.   ,  0.   , -0.001,  0.   , -0.019,  0.014,  0.267,  1.   ]])
        >>> result3 = model3.analyze(exmatrix=exmat, result=resmat, sigma=sigma, add_const=True)
        >>> result3.params.round(2)
        array([-1.64,  1.89,  1.  ,  1.78,  2.16,  2.6 ])
        >>> result3.resid.round(2)
        array([ 0.23, -0.02, -0.29,  0.18, -0.32,  0.23,  0.23, -0.04])
        >>> result3.predict().round(2)
        array([2.74, 0.26, 1.97, 3.03, 2.3 , 5.02, 1.52, 7.79])
        """
        if not isinstance(exmatrix, np.ndarray):
            raise TypeError(f"exmatrix expected numpy.ndarray; got {type(exmatrix)}")
        if not isinstance(result, np.ndarray):
            raise TypeError(f"result expected numpy.ndarray; got {type(result)}")
        assert exmatrix.shape[0] == result.shape[0], \
            f"numbers of rows in exmatrix & result should be equal; \
                got {exmatrix.shape[0]} & {result.shape[0]}"
        args = [
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
        ret_model = self.model[1](**args, **kwargs)
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
