'''
_Analyser Class of One-way ANOVA
'''

import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm

from statsmodels.formula.api import ols
from tagupy.type import _Analyzer as Analyzer
from tagupy.utils import is_correct_id_list, is_int_2d_array
from typing import List, NamedTuple


class OAResult(NamedTuple):
    '''
    NamedTuple Class of One-way ANOVA Results
    '''
    exmatrix: np.ndarray
    resmatrix: np.ndarray
    model: List[List[statsmodels.regression.linear_model.RegressionResultsWrapper]]
    table: list[np.ndarray, dict[str, List[str]]]


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
        None

    def _input_proc(
        self,
        exmatrix: np.ndarray,
        resmatrix: np.ndarray,
        factor: int,
        result_id: List[int]
    ) -> pd.core.frame.DataFrame:
        '''
        process the data into the appropriate form

        Parameters
        ----------
        exmatrix: numpy.ndarray(number_of_runs, number_of_factors)
            experiment matrix
        resmatrix: numpy.ndarray(number_of_runs, number_of_experiments)
            result matrix
        factor_id: int
            index of factor expected to be analyzed
        result_id: List[int]
            list of results indices expected to be analyzed
        Return
        ------
        df_data: pd.core.frame.DataFrame
        '''
        y_id = [f'y{i}' for i in result_id]
        ele, count = np.unique(exmatrix[:, factor], return_counts=True)
        temp = [[e, c]for e, c in zip(ele, count)]
        result_bool = [i in result_id for i in range(resmatrix.shape[1])]
        y_l = [resmatrix[exmatrix[:, factor] == 1][:, result_bool] for i in temp]
        data = [np.concatenate((y, np.full((i[1], 1), i[0])), axis=1) for y, i in zip(y_l, temp)]
        df_data = pd.DataFrame(np.concatenate(data), columns=[*y_id, 'group'])
        df_data['group'] = df_data['group'].astype(str)

        return df_data

    def analyze(
        self,
        exmatrix: np.ndarray,
        resmatrix: np.ndarray,
        factor_id: List[int] = [],
        result_id: List[int] = [],
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
            list of indices (>=0) of factors expected to be analyzed
        result_id: List[int]
            list of indices (>=0) of experimnets expected to be analyzed

        Return
        ------
        analysis_result: NamedTuple
            Result of analysis method
            exmatrix: np.ndarray,
            resmatrix: np.ndarray,
            model: List[List[statsmodels.regression.linear_model.RegressionResultsWrapper]]
                list of created models of all the factor and the experiment
            table: List[np.ndarray, dict[str, List[str]]]
                list contains One_way ANOVA table, its index name, and its column name

        Note
        ----
        elements of factor_id, result_id expected to be non negative integer for implementation

        Example
        -------
        >>> import numpy as np
        >>> from tagupy.design.analyzer import OnewayANOVA
        >>> analyzer = OnewayANOVA()
        >>> result = analyzer.analyze(
        ...     np.array([[-1, -1,  1, -1,  1,  1],
        ...               [ 0,  1,  1,  1,  1,  1],
        ...               [ 0, -1, -1, -1, -1, -1],
        ...               [ 1, -1, -1,  0,  1,  1],
        ...               [ 1, -1,  1, -1, -1,  0],
        ...               [ 1,  1,  1, -1,  1, -1],
        ...               [-1, -1, -1,  1, -1,  1],
        ...               [ 1,  0,  1,  1, -1,  1],
        ...               [-1,  1,  0, -1, -1,  1],
        ...               [ 1,  1, -1,  1, -1, -1],
        ...               [-1,  1,  1,  0, -1, -1],
        ...               [-1,  1, -1,  1,  1,  0],
        ...               [-1,  0, -1, -1,  1, -1],
        ...               [-1, -1,  1,  1,  0, -1],
        ...               [ 1,  1, -1, -1,  0,  1],
        ...               [ 1, -1,  0,  1,  1, -1],
        ...               [ 0,  0,  0,  0,  0,  0]]),
        ...     np.array([[0.63019297, 0.58619756, 0.43411374],
        ...               [0.91747724, 0.32393429, 0.04366332],
        ...               [0.21524266, 0.81815103, 0.40212519],
        ...               [0.88389254, 0.25744192, 0.03066148],
        ...               [0.59410509, 0.96162365, 0.62316969],
        ...               [0.56927398, 0.39596578, 0.76887992],
        ...               [0.4968997 , 0.35014733, 0.74177565],
        ...               [0.06223279, 0.17630325, 0.67672164],
        ...               [0.63636881, 0.25071343, 0.55393937],
        ...               [0.38908674, 0.59377007, 0.18133165],
        ...               [0.06753605, 0.71909372, 0.67057174],
        ...               [0.73643459, 0.25253943, 0.982259  ],
        ...               [0.88522878, 0.37885563, 0.69005722],
        ...               [0.63432006, 0.93461394, 0.80317958],
        ...               [0.13611039, 0.54638813, 0.0495983 ],
        ...               [0.57250679, 0.61633224, 0.52158236],
        ...               [0.91144691, 0.64098009, 0.17661229]]),
        ... )

        >>> result.exmatrix
        array([[-1, -1,  1, -1,  1,  1],
               [ 0,  1,  1,  1,  1,  1],
               [ 0, -1, -1, -1, -1, -1],
               [ 1, -1, -1,  0,  1,  1],
               [ 1, -1,  1, -1, -1,  0],
               [ 1,  1,  1, -1,  1, -1],
               [-1, -1, -1,  1, -1,  1],
               [ 1,  0,  1,  1, -1,  1],
               [-1,  1,  0, -1, -1,  1],
               [ 1,  1, -1,  1, -1, -1],
               [-1,  1,  1,  0, -1, -1],
               [-1,  1, -1,  1,  1,  0],
               [-1,  0, -1, -1,  1, -1],
               [-1, -1,  1,  1,  0, -1],
               [ 1,  1, -1, -1,  0,  1],
               [ 1, -1,  0,  1,  1, -1],
               [ 0,  0,  0,  0,  0,  0]])
        
        >>> result.resmatrix
        array([[0.63019297, 0.58619756, 0.43411374],
               [0.91747724, 0.32393429, 0.04366332],
               [0.21524266, 0.81815103, 0.40212519],
               [0.88389254, 0.25744192, 0.03066148],
               [0.59410509, 0.96162365, 0.62316969],
               [0.56927398, 0.39596578, 0.76887992],
               [0.4968997 , 0.35014733, 0.74177565],
               [0.06223279, 0.17630325, 0.67672164],
               [0.63636881, 0.25071343, 0.55393937],
               [0.38908674, 0.59377007, 0.18133165],
               [0.73643459, 0.25253943, 0.982259  ],
               [0.88522878, 0.37885563, 0.69005722],
               [0.63432006, 0.93461394, 0.80317958],
               [0.13611039, 0.54638813, 0.0495983 ],
               [0.57250679, 0.61633224, 0.52158236],
               [0.91144691, 0.64098009, 0.17661229]])
        
        >>> result.models
        [[<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b662bfa00>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66d1da90>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66fbd6d0>],
         [<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66d2cee0>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66d33a30>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66d3d820>],
         [<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66d42850>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66f51cd0>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66d46490>],
         [<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66c6fd60>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66c85730>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66c857f0>],
         [<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66991ac0>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66868d90>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b66878b50>],
         [<statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b666dc8b0>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b666d7ac0>,
          <statsmodels.regression.linear_model.RegressionResultsWrapper at 0x7f5b666d7a90>]]

        >>> result.table[0]
        array([[2.00000000e+00, 1.18829915e-01, 5.94149576e-02, 6.86278716e-01,
                5.19605070e-01, 0.00000000e+00, 0.00000000e+00],
               [1.40000000e+01, 1.21205771e+00, 8.65755505e-02,            nan,
                           nan, 0.00000000e+00, 0.00000000e+00],
               [2.00000000e+00, 2.17437755e-02, 1.08718877e-02, 1.61613374e-01,
                8.52334719e-01, 1.00000000e+00, 0.00000000e+00],
               [1.40000000e+01, 9.41793519e-01, 6.72709656e-02,            nan,
                           nan, 1.00000000e+00, 0.00000000e+00],
               [2.00000000e+00, 5.85845503e-01, 2.92922752e-01, 4.91419378e+00,
                2.41676077e-02, 2.00000000e+00, 0.00000000e+00],
               [1.40000000e+01, 8.34504845e-01, 5.96074889e-02,            nan,
                           nan, 2.00000000e+00, 0.00000000e+00],
               [2.00000000e+00, 4.16196925e-02, 2.08098463e-02, 2.25971531e-01,
                8.00594416e-01, 0.00000000e+00, 1.00000000e+00],
               [1.40000000e+01, 1.28926793e+00, 9.20905664e-02,            nan,
                           nan, 0.00000000e+00, 1.00000000e+00],
               [2.00000000e+00, 2.00232189e-01, 1.00116094e-01, 1.83625828e+00,
                1.95797647e-01, 1.00000000e+00, 1.00000000e+00],
               [1.40000000e+01, 7.63305106e-01, 5.45217933e-02,            nan,
                           nan, 1.00000000e+00, 1.00000000e+00],
               [2.00000000e+00, 8.67731507e-03, 4.33865754e-03, 4.30278146e-02,
                9.58010911e-01, 2.00000000e+00, 1.00000000e+00],
               [1.40000000e+01, 1.41167303e+00, 1.00833788e-01,            nan,
                           nan, 2.00000000e+00, 1.00000000e+00],
               [2.00000000e+00, 9.54397035e-02, 4.77198518e-02, 5.40757659e-01,
                5.93993090e-01, 0.00000000e+00, 2.00000000e+00],
               [1.40000000e+01, 1.23544792e+00, 8.82462799e-02,            nan,
                           nan, 0.00000000e+00, 2.00000000e+00],
               [2.00000000e+00, 5.87498162e-02, 2.93749081e-02, 4.54525204e-01,
                6.43793174e-01, 1.00000000e+00, 2.00000000e+00],
               [1.40000000e+01, 9.04787478e-01, 6.46276770e-02,            nan,
                           nan, 1.00000000e+00, 2.00000000e+00],
               [2.00000000e+00, 8.32966867e-02, 4.16483434e-02, 4.36090805e-01,
                6.55048549e-01, 2.00000000e+00, 2.00000000e+00],
               [1.40000000e+01, 1.33705366e+00, 9.55038330e-02,            nan,
                           nan, 2.00000000e+00, 2.00000000e+00],
               [2.00000000e+00, 2.01471196e-02, 1.00735598e-02, 1.07595544e-01,
                8.98726061e-01, 0.00000000e+00, 3.00000000e+00],
               [1.40000000e+01, 1.31074050e+00, 9.36243216e-02,            nan,
                           nan, 0.00000000e+00, 3.00000000e+00],
               [2.00000000e+00, 3.56920244e-02, 1.78460122e-02, 2.69273530e-01,
                7.67801938e-01, 1.00000000e+00, 3.00000000e+00],
               [1.40000000e+01, 9.27845270e-01, 6.62746622e-02,            nan,
                           nan, 1.00000000e+00, 3.00000000e+00],
               [2.00000000e+00, 1.56769145e-01, 7.83845725e-02, 8.68471304e-01,
                4.41015367e-01, 2.00000000e+00, 3.00000000e+00],
               [1.40000000e+01, 1.26358120e+00, 9.02558002e-02,            nan,
                           nan, 2.00000000e+00, 3.00000000e+00],
               [2.00000000e+00, 5.34195611e-01, 2.67097805e-01, 4.69361965e+00,
                2.75450481e-02, 0.00000000e+00, 4.00000000e+00],
               [1.40000000e+01, 7.96692011e-01, 5.69065722e-02,            nan,
                           nan, 0.00000000e+00, 4.00000000e+00],
               [2.00000000e+00, 2.10852602e-01, 1.05426301e-01, 1.96093827e+00,
                1.77505527e-01, 1.00000000e+00, 4.00000000e+00],
               [1.40000000e+01, 7.52684692e-01, 5.37631923e-02,            nan,
                           nan, 1.00000000e+00, 4.00000000e+00],
               [2.00000000e+00, 9.00871395e-02, 4.50435698e-02, 4.74049025e-01,
                6.32112946e-01, 2.00000000e+00, 4.00000000e+00],
               [1.40000000e+01, 1.33026321e+00, 9.50188006e-02,            nan,
                           nan, 2.00000000e+00, 4.00000000e+00],
               [2.00000000e+00, 1.56040354e-01, 7.80201768e-02, 9.29722956e-01,
                4.17715104e-01, 0.00000000e+00, 5.00000000e+00],
               [1.40000000e+01, 1.17484727e+00, 8.39176620e-02,            nan,
                           nan, 0.00000000e+00, 5.00000000e+00],
               [2.00000000e+00, 3.12819793e-01, 1.56409897e-01, 3.36511397e+00,
                6.40719155e-02, 1.00000000e+00, 5.00000000e+00],
               [1.40000000e+01, 6.50717501e-01, 4.64798215e-02,            nan,
                           nan, 1.00000000e+00, 5.00000000e+00],
               [2.00000000e+00, 2.00786826e-01, 1.00393413e-01, 1.15246787e+00,
                3.44082518e-01, 2.00000000e+00, 5.00000000e+00],
               [1.40000000e+01, 1.21956352e+00, 8.71116802e-02,            nan,
                           nan, 2.00000000e+00, 5.00000000e+00]])

        >>> result.table[1]['index']
        ['group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual',
         'group',
         'Residual']

        >>> result.table[1]['column']
        ['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)', 'result_id', 'factor_id']
        '''
        assert is_int_2d_array(exmatrix), \
            f'exmatrix expected a 2d matrix in which all elements are integer. \nGot: {exmatrix}'
        assert isinstance(resmatrix, np.ndarray), \
            f'resmatrix expected a numpy.ndarray, got: {type(resmatrix)}'
        n_fac = exmatrix.shape[1]
        n_res = resmatrix.shape[1]
        if factor_id == []:
            factor_id = [i for i in range(n_fac)]
        if result_id == []:
            result_id = [i for i in range(n_res)]
        assert is_correct_id_list(factor_id, exmatrix), \
            f'factor_id got incorrect input {factor_id} see the requirements described in the api.'
        assert is_correct_id_list(result_id, exmatrix), \
            f'result_id got incorrect input {result_id} see the requirements described in the api.'  
        t_list = []
        model = []
        for idx, fac in enumerate(factor_id):
            data = self._input_proc(exmatrix, resmatrix, fac, result_id)
            m = [ols(f'y{i} ~ group', data=data).fit() for i in result_id]
            t = pd.concat([sm.stats.anova_lm(i, typ=1) for i in m])
            t['result_id'] = sum([[i, i] for i in result_id], [])
            t['factor_id'] = [idx] * t.shape[0]
            model.append(m)
            t_list.append(t)
        res = (pd.concat(t_list))

        return OAResult(
            exmatrix=exmatrix,
            resmatrix=resmatrix,
            model=model,
            table=[res.to_numpy(), {'index': list(res.index), 'column': list(res.columns)}]
        )
