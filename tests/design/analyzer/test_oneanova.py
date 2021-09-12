import pytest
import numpy as np

from tagupy.design.analyzer import OnewayANOVA


@pytest.fixture
def valid_matrix_input():
    return{
        'dsd': {
            'exmatrix': np.array([[-1, -1,  1, -1,  1,  1],
                                  [0,  1,  1,  1,  1,  1],
                                  [0, -1, -1, -1, -1, -1],
                                  [1, -1, -1,  0,  1,  1],
                                  [1, -1,  1, -1, -1,  0],
                                  [1,  1,  1, -1,  1, -1],
                                  [-1, -1, -1,  1, -1,  1],
                                  [1,  0,  1,  1, -1,  1],
                                  [-1,  1,  0, -1, -1,  1],
                                  [1,  1, -1,  1, -1, -1],
                                  [-1,  1,  1,  0, -1, -1],
                                  [-1,  1, -1,  1,  1,  0],
                                  [-1,  0, -1, -1,  1, -1],
                                  [-1, -1,  1,  1,  0, -1],
                                  [1,  1, -1, -1,  0,  1],
                                  [1, -1,  0,  1,  1, -1],
                                  [0,  0,  0,  0,  0,  0]]),
            'resmatrix': np.array([[0.63019297, 0.58619756, 0.43411374],
                                  [0.91747724, 0.32393429, 0.04366332],
                                  [0.21524266, 0.81815103, 0.40212519],
                                  [0.88389254, 0.25744192, 0.03066148],
                                  [0.59410509, 0.96162365, 0.62316969],
                                  [0.56927398, 0.39596578, 0.76887992],
                                  [0.4968997, 0.35014733, 0.74177565],
                                  [0.06223279, 0.17630325, 0.67672164],
                                  [0.63636881, 0.25071343, 0.55393937],
                                  [0.38908674, 0.59377007, 0.18133165],
                                  [0.06753605, 0.71909372, 0.67057174],
                                  [0.73643459, 0.25253943, 0.982259],
                                  [0.88522878, 0.37885563, 0.69005722],
                                  [0.63432006, 0.93461394, 0.80317958],
                                  [0.13611039, 0.54638813, 0.0495983],
                                  [0.57250679, 0.61633224, 0.52158236],
                                  [0.91144691, 0.64098009, 0.17661229]])},
        'fullfact': {
            'exmatrix': np.array([[1, 0, 1],
                                  [0, 2, 1],
                                  [0, 1, 1],
                                  [1, 1, 1],
                                  [0, 1, 0],
                                  [0, 0, 1],
                                  [1, 0, 0],
                                  [1, 1, 0],
                                  [0, 0, 0],
                                  [0, 2, 0],
                                  [1, 2, 1],
                                  [1, 2, 0]]),
            'resmatrix': np.array([[0.54932946, 0.54434416, 0.45185261],
                                   [0.01448972, 0.15018607, 0.56758295],
                                   [0.73700129, 0.7164212, 0.3687976],
                                   [0.73634662, 0.59285513, 0.21240577],
                                   [0.21033471, 0.09348808, 0.06756687],
                                   [0.25697798, 0.92748727, 0.1818888],
                                   [0.16740076, 0.2949764, 0.21614899],
                                   [0.16966721, 0.34482373, 0.36390886],
                                   [0.55732665, 0.98882151, 0.04974632],
                                   [0.67286307, 0.82310922, 0.98649674],
                                   [0.76581243, 0.80444864, 0.42049911],
                                   [0.46511951, 0.04133385, 0.19324854]])},
        'pb': {
            'exmatrix': np.array([[1, -1, -1, -1,  1, -1, -1,  1, -1,  1],
                                  [1,  1,  1, -1, -1, -1,  1, -1, -1,  1],
                                  [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
                                  [1, -1, -1,  1, -1,  1,  1,  1, -1, -1],
                                  [-1, -1,  1, -1,  1,  1,  1, -1, -1, -1],
                                  [1, -1,  1,  1,  1, -1, -1, -1,  1, -1],
                                  [-1, -1, -1,  1, -1, -1,  1, -1,  1,  1],
                                  [-1,  1,  1,  1, -1, -1, -1,  1, -1, -1],
                                  [-1,  1, -1, -1,  1, -1,  1,  1,  1, -1],
                                  [-1, -1,  1, -1, -1,  1, -1,  1,  1,  1],
                                  [1,  1, -1, -1, -1,  1, -1, -1,  1, -1],
                                  [-1,  1, -1,  1,  1,  1, -1, -1, -1,  1]]),
            'resmatrix': np.array([[0.13821059, 0.1395963, 0.71263942],
                                   [0.89103128, 0.02298159, 0.70940104],
                                   [0.36454655, 0.74518391, 0.26008459],
                                   [0.10175229, 0.96904007, 0.36044816],
                                   [0.23082782, 0.67977631, 0.24459614],
                                   [0.38541218, 0.74437929, 0.8264849],
                                   [0.87947899, 0.08342015, 0.91155078],
                                   [0.75348228, 0.4164176, 0.72325453],
                                   [0.22694426, 0.46871851, 0.98292845],
                                   [0.85554857, 0.98629489, 0.97601873],
                                   [0.37473258, 0.89093493, 0.93733872],
                                   [0.38686135, 0.3245964, 0.42791052]])}
        }


@pytest.fixture
def valid_mock_input():
    return {
        'exmatrix': np.ones((4, 4), dtype=np.int64),
        'resmatrix': np.ones((4, 3)),
    }


@pytest.fixture
def invalid_exmatrix_value_input():
    return {
        'exmatrix': np.full((4, 4), 1.2),
        'resmatrix': np.ones((4, 3)),
    }


@pytest.fixture
def invalid_exmatrix_shape_input():
    return {
        'exmatrix': np.full((4, 4, 3), 1),
        'resmatrix': np.ones((4, 3)),
    }


@pytest.fixture
def invalid_resmatrix_input():
    return {
        'exmatrix': np.full((4, 4), 1),
        'resmatrix': [[1, 2, 3],
                      [1, 2, 3]]
    }


@pytest.fixture
def valid_id_input():
    return [0, 1]


@pytest.fixture
def invalid_id_input():
    return[
        np.zeros((3)),
        [-1, 1, 2],
        [1, 1, 1],
        [1, 2, 4],
    ]


# green
def test_analyze_valid_input(valid_matrix_input, valid_id_input):
    for i in ['dsd', 'fullfact', 'pb']:
        analysis = OnewayANOVA()
        result = analysis.analyze(
            **valid_matrix_input[i],
            factor_id=valid_id_input,
            result_id=valid_id_input,
        )
        exmat = valid_matrix_input[i]['exmatrix']
        resmat = valid_matrix_input[i]['resmatrix']
        assert np.array_equal(result.exmatrix, exmat), \
            f'result.exmatrix expected {exmat}, got: {result.exmatrix}'
        assert np.array_equal(result.resmatrix, resmat), \
            f'result.resmatrix expected {resmat}, got: {result.resmatrix}'
        assert isinstance(result.model, list), \
            f'type of result.model expected list, got: {type(result.model)}'
        assert isinstance(result.table['table'], np.ndarray), \
            f'result.table["table"] expected 2d np.ndarray, got {result.table["table"]}'


# red
def test_analyze_invalid_input_exmatrix(
        invalid_exmatrix_value_input,
        invalid_exmatrix_shape_input,
        valid_id_input):

    analysis1 = OnewayANOVA()

    with pytest.raises(AssertionError) as e:
        analysis1.analyze(
            **invalid_exmatrix_value_input,
            factor_id=valid_id_input,
            result_id=valid_id_input,
        )

    assert f'{invalid_exmatrix_value_input["exmatrix"]}' in f'{e.value}', \
        f'Assertion message should contain reasons, got {e.value}'

    analysis2 = OnewayANOVA()

    with pytest.raises(AssertionError) as e:
        analysis2.analyze(
            **invalid_exmatrix_shape_input,
            factor_id=valid_id_input,
            result_id=valid_id_input,
        )

    assert f'{invalid_exmatrix_shape_input["exmatrix"]}' in f'{e.value}', \
        f'Assertion message should contain reasons, got {e.value}'


def test_analyze_invalid_input_resmatrix(
        invalid_resmatrix_input,
        valid_id_input):

    analysis = OnewayANOVA()

    with pytest.raises(AssertionError) as e:
        analysis.analyze(
            **invalid_resmatrix_input,
            factor_id=valid_id_input,
            result_id=valid_id_input
        )

    assert f'{type(invalid_resmatrix_input["resmatrix"])}' in f'{e.value}', \
        f'Assertion message should contain reasons, got {e.value}'


def test_analyze_invalid_input_id(
        valid_mock_input,
        valid_id_input,
        invalid_id_input):

    for id, k in enumerate(invalid_id_input):
        analysis = OnewayANOVA()

        with pytest.raises(AssertionError) as e:
            analysis.analyze(
                **valid_mock_input,
                factor_id=k,
                result_id=valid_id_input
            )
        if id == 0 or id == 1:
            assert 'list of' in f'{e.value}', \
                f'Assertation message should contain reasons,expexted got {e.value}'
        elif id == 2:
            assert 'duplicated' in f'{e.value}', \
                f'Assertation message should contain reasons, got {e.value}'
        else:
            assert 'out of range' in f'{e.value}', \
                f'Assertation message should contain reasons, got {e.value}'
