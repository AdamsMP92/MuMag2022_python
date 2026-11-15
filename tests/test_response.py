import numpy as np

from mumag2022 import ExperimentGeometry, characteristic_q_max_nm_inv, response_functions


def test_parallel_response_matches_closed_form():
    q = np.array([0.05, 0.1])
    h0 = np.array([100.0, 200.0])
    ms = np.array([1000.0, 1000.0])
    hd = np.array([20.0, 20.0])
    response, magnetostatic = response_functions(
        q, h0, ms, hd, 5.0, ExperimentGeometry.PARALLEL
    )
    assert response.shape == (2, 2)
    assert magnetostatic is None
    assert np.all(response > 0.0)
    assert np.all(response[1] < response[0])


def test_perpendicular_responses_decrease_with_field_and_q():
    response_h, response_m = response_functions(
        [0.03, 0.15],
        [100.0, 500.0],
        [1200.0, 1200.0],
        [10.0, 10.0],
        7.0,
        ExperimentGeometry.PERPENDICULAR,
    )
    assert np.all(response_h[:, 1] < response_h[:, 0])
    assert np.all(response_m[:, 1] < response_m[:, 0])
    assert np.all(response_h[1] < response_h[0])
    assert np.all(response_m[1] < response_m[0])


def test_characteristic_q_max_is_positive_and_decreases_with_exchange():
    low_a = characteristic_q_max_nm_inv(5.0, 1200.0, 1000.0)
    high_a = characteristic_q_max_nm_inv(10.0, 1200.0, 1000.0)
    assert low_a > high_a > 0.0
