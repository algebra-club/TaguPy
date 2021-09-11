'''
_Analyser Class of One-way ANOVA
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.formula.api import ols
from tagupy.type import _Analyzer as Analyzer
from typing import List, NamedTuple


class OAResult(NamedTuple):
    '''
    NamedTuple Class of One-way ANOVA Results
    '''
    exmatrix: np.ndarray
    resmatrix: np.ndarray
    model: List[List[statsmodels.regression.linear_model.RegressionResultsWrapper]]
    table: List[np.ndarray, List[str]]


class OnewayANOVA(Analyzer):
    '''
    Analyzer Module of One-way ANOVA

    Method
    ------
    analyze(exmatrix: np.ndarray, result:np.ndarray) -> NamedTuple

    Notes
    -----
    see also:
    https://www.statsmodels.org/stable/anova.html
    '''
    def __init__(self):

    def _input_proc(
        self,
        exmatrix: np.ndarray,
        resmatrix: np.ndarray,
        factor_id: List[int],
    ) -> pd.core.frame.DataFrame:
        '''
        process the data into the appropriate form

        Parameters
        ----------
        exmatrix: numpy.ndarray(number_of_runs, number_of_factors)
            experiment matrix
        resmatrix: numpy.ndarray(number_of_runs, number_of_experiments)
            result matrix
        factor_id: List[int]
            list of indices of factor expected to be analyzed
       
        Return
        ------
        df_data: pd.core.frame.DataFrame
        '''
        n_res = resmatrix.shape[1]
        y_id = [f'y{i}' for i in range(n_res)]
        ele, count = np.unique(exmatrix[:, factor_id], return_counts=True)
        temp = [[e, c]for e, c in zip(ele, count)]
        y_l = [resmatrix[exmatrix[:, factor_id] == i[0]] for i in temp]
        data = [np.concatenate((y, np.full((i[1], 1), i[0])), axis=1) for y, i in zip(y_l, temp)]
        df_data = pd.DataFrame(np.concatenate(temp), columns=[*y_id, 'group'])
        df_data['group'] = df_data['group'].astype(str)
        return df_data

    def analyze(
        self,
        exmatrix: np.ndarray,
        resmatrix: np.ndarray,
        factor_id: List[int] = [:],
        result_id: List[int] = [:],
    ) -> NamedTuple:
        '''
        return One-way ANOVA table

        Parameters
        ----------
        exmatrix: numpy.ndarray(number_of_runs, number_of_factors)
            experiment matrix
        resmatrix: numpy.ndarray(number_of_runs, number_of_experiments)
            result matrix
        factor_id: List[int]
            list of indices of factor expected to be analyzed
        result_id: List[int]
            list of indices of experimnets expected to be analyzed

        return
        ------
        analysis_result: NamedTuple
            Result of analysis method
            exmatrix: np.ndarray,
            resmatrix: np.ndarray,
            model: List[List[statsmodels.regression.linear_model.RegressionResultsWrapper]]
                list of created models of all the factor and the experiment
            table: List[numpy.ndarray, List[str]]
                list includes np.ndarray of One-way ANOVA table and its column name
        '''
        table = []
        model = []
        for fac_id in range(exmatrix.shape[1]): 
            data = _input_proc(exmatrix=exmatrix, resmatrix=resmatrix, factor_id=factor_id)
            m = [ols(f'y{i} ~ group', data=data).fit() for i in range(n_res)]
            t = pd.concat([sm.stats.anova_lm(i, typ=1) for i in m])
            model.append(m)
            table.append(t)
            result = (pd.concat(table))

        return OAResult(
            exmatrix=exmatrix,
            resmatrix=resmatrix,
            model=model,
            table=
        )
