'''
Test for Linear Analysis Module
'''

import numpy as np
import pytest
import statsmodels.api as sm
from tagupy.design.analyzer import LinearAnalysis
from tagupy.design.analyzer._la import LAResult
from typing import Callable


@pytest.fixture
def correct_input():
    random_tuple = np.random.randint(2, 100, (1, 2)).tolist()[0]
    size = (max(random_tuple), min(random_tuple))
    return {
            "exmatrix": np.random.randint(-100, 100, size),
            "result": np.random.randn(size[0], 1)
    }


def test_init_invalid_input():
    arg = [0, 4.5, np.array([1, 1, 1]), None, "hoge"]
    for v in arg:
        with pytest.raises(AssertionError) as e:
            LinearAnalysis(v)
        assert f"{v}" in f"{e.value}", \
            f"NoReasons: Inform the AssertionError reasons, got {e.value}"


def test_init_correct_input():
    arg = ["OLS", "WLS", "GLS"]
    exp = [sm.OLS, sm.WLS, sm.GLS]
    for i, v in enumerate(arg):
        model = LinearAnalysis(v)
        assert model.model[1] == exp[i], \
            f"self.model expected to contain {v}, got {model.model}"


def test_analyze_invalid_input_matrix():
    arg = [
        # invalid dtype
        {"exmatrix": "hoge", "result": "moge"},
        {"exmatrix": 1, "result": 16},
        {"exmatrix": 3.4, "result": -3.5},
        {"exmatrix": np.ones((4, 5), dtype=int), "result": "moge"},
        {"exmatrix": np.ones((4, 5), dtype=int), "result": 16},
        {"exmatrix": np.ones((4, 5), dtype=int), "result": -3.5},
        {"exmatrix": "hoge", "result": np.ones((5, 12))},
        {"exmatrix": 1, "result": np.ones((5, 12))},
        {"exmatrix": 3.4, "result": np.ones((5, 12))},
        # inconsistency in number of rows
        {"exmatrix": np.ones((4, 5), dtype=int), "result": np.ones((1, 2))},
        {"exmatrix": np.ones((4, 5), dtype=int), "result": np.ones((5, 4))},
        {"exmatrix": np.ones((4, 5), dtype=int), "result": np.ones((11, 12))},
        {"exmatrix": np.ones((17, 2), dtype=int), "result": np.ones((1, 2))},
        {"exmatrix": np.ones((17, 2), dtype=int), "result": np.ones((2, 17))},
        {"exmatrix": np.ones((17, 2), dtype=int), "result": np.ones((46, 46))},
    ]
    for v in arg[:9]:
        model_ols = LinearAnalysis("OLS")
        model_wls = LinearAnalysis("WLS")
        model_gls = LinearAnalysis("GLS")
        with pytest.raises(TypeError) as e_type:
            model_ols.analyze(**v)
            model_wls.analyze(**v)
            model_gls(**v)
        type_exmat = f"{type(v['exmatrix'])}" in f"{e_type.value}"
        type_res = f"{type(v['result'])}" in f"{e_type.value}"
        assert type_exmat or type_res, \
            f"NoReasons: Inform the TypeError reasons, got {e_type.value}"
    for v in arg[9:]:
        model_ols = LinearAnalysis("OLS")
        model_wls = LinearAnalysis("WLS")
        model_gls = LinearAnalysis("GLS")
        with pytest.raises(AssertionError) as e_shape:
            model_ols.analyze(**v)
            model_wls.analyze(**v)
            model_gls(**v)
        shape_exmat = f"{v['exmatrix'].shape[0]}" in f"{e_shape.value}"
        shape_res = f"{v['result'].shape[0]}" in f"{e_shape.value}"
        assert shape_exmat and shape_res, \
            f"NoReasons: Inform the AssertionError reasons, got {e_shape.value}"


def test_analyze_correct_input_matrix(correct_input):
    l_models = [LinearAnalysis("OLS"), LinearAnalysis("WLS"), LinearAnalysis("GLS")]
    for model in l_models:
        ret = model.analyze(**correct_input)
        assert isinstance(ret, LAResult), \
            f"dtype of return value expected LAResult; got {type(ret)}"
        assert (ret.exmatrix == correct_input["exmatrix"]).all(), \
            f"self.exmatrix should be equal to exmatrix; got {ret.exmatrix}"
        assert (ret.resmatrix == correct_input["result"]).all, \
            f"self.resmatrix should be equal to relust; got {ret.resmatrix}"


def test_analyze_correct_return(correct_input):
    l_models = [LinearAnalysis("OLS"), LinearAnalysis("WLS"), LinearAnalysis("GLS")]
    l_exp = [sm.OLS, sm.WLS, sm.GLS]
    for model, exp in zip(l_models, l_exp):
        ret = model.analyze(**correct_input, add_const=False)
        res_exp = exp(endog=correct_input["result"], exog=correct_input["exmatrix"]).fit()
        # dtype of instance (except self.model) in LAResult
        assert isinstance(ret.params, np.ndarray), \
            f"self.params expected numpy.ndarray; got {type(ret.params)}"
        assert isinstance(ret.resid, np.ndarray), \
            f"self.resid expected numpy.ndarray; got {type(ret.resid)}"
        assert isinstance(ret.bse, np.ndarray), \
            f"self.bse expected numpy.ndarray; got {type(ret.bse)}"
        assert isinstance(ret.predict, Callable), \
            f"self.predict expected Callable; got {type(ret.predict)}"
        assert isinstance(ret.rsquared, float), \
            f"self.rsquared expected float; got {type(ret.rsquared)}"
        assert isinstance(ret.summary, Callable), \
            f"self.summary expected Callable; got {type(ret.summary)}"
        # equal values as statsmodels
        l_v_ret = [
            ret.params,
            ret.resid,
            ret.bse,
            ret.predict(),
            ret.rsquared
        ]
        l_v_model = [
            # for self.model
            ret.model.params,
            ret.model.resid,
            ret.model.bse,
            ret.model.predict(),
            ret.model.rsquared,
        ]
        l_v_exp = [
            res_exp.params,
            res_exp.resid,
            res_exp.bse,
            res_exp.predict(),
            res_exp.rsquared
        ]
        l_instance = [
            "params",
            "resid",
            "bse",
            "predict()",
            "rsquared"
        ]
        for v_ret, v_model, v_exp, v_i in zip(l_v_ret, l_v_model, l_v_exp, l_instance):
            # self.model
            if isinstance(v_model, np.ndarray):
                assert (v_model == v_exp).all(), \
                    f"self.model.{v_i} should be equal to the result of statsmodels ({v_exp}); \
                        got {v_model}"
            else:
                assert v_model == v_exp, \
                    f"self.model.{v_i} should be equal to the result of statsmodels ({v_exp}); \
                        got {v_model}"
            # other instances
            if isinstance(v_ret, np.ndarray):
                assert (v_ret == v_exp).all(), \
                    f"self.{v_i} should be equal to the result of statsmodels ({v_exp}); \
                        got {v_ret}"
            else:
                assert all([v_ret == v_exp]), \
                    f"self.{v_i} should be equal to the result of statsmodels ({v_exp}); \
                        got {v_ret}"
