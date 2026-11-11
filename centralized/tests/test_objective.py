"""Objective terms against a hand-computed instance.

Instance: 1 EV, T=4, dt=1 h, prices [1,2,3,4], C_ref=1, P_ref=10, H_ref=1,
B=10 kWh, SOC 0 -> 0.5 (R=5), eta=1, plugged in t=0..3, P=[2,1,0,0].
  J_cost  = 1*2 + 2*1                       = 4
  J_peak  = (0.2^2 + 0.1^2 + 0 + 0) / 4      = 0.0125
  J_delay = (0*2 + 1*1) / 5                  = 0.2
  s = 5 - 3 = 2,  J_fair = (2/5)^2           = 0.16
"""
import numpy as np
import pytest

from common.objective import (Weights, linear_coefficients, objective_components,
                              optimal_slack, prox_peak, peak_quadratic_coeff, total_objective)
from common.utils import ProblemData, random_instance


@pytest.fixture
def tiny():
    data = ProblemData(T=4, dt=1.0, prices=[1, 2, 3, 4], grid_cap=100, station_cap=[100],
                       station=[0], battery=[10], soc_init=[0.0], soc_target=[0.5],
                       eta=[1.0], p_max=[5], arrival=[0], departure=[4],
                       C_ref=1.0, P_ref=10.0, H_ref=1.0)
    return data, np.array([[2.0, 1.0, 0.0, 0.0]])


def test_hand_computed(tiny):
    data, P = tiny
    s = optimal_slack(data, P)
    assert s[0] == pytest.approx(2.0)
    c = objective_components(data, P, s)
    assert c["J_cost"] == pytest.approx(4.0)
    assert c["J_peak"] == pytest.approx(0.0125)
    assert c["J_delay"] == pytest.approx(1.8)
    assert c["J_fair"] == pytest.approx(0.56)
    assert total_objective(data, P, s, Weights(2, 3, 4)) == pytest.approx(4 + 2 * 0.0125 + 3 * 1.8 + 4 * 0.56)


def test_linear_coefficients_identity():
    data = random_instance(15, 2, seed=1)
    P = np.random.default_rng(0).uniform(0, 1, (data.N, data.T)) * data.p_upper
    w = Weights(1, 3, 1)
    c = objective_components(data, P)
    assert np.sum(linear_coefficients(data, w) * P) == pytest.approx(c["J_cost"] + 3 * (c["J_delay"] - data.delay_const))

def test_cvxpy_components_equal_numpy():
    cp = pytest.importorskip("cvxpy")
    from common.objective import cvx_components
    data = random_instance(10, 2, seed=2)
    rng = np.random.default_rng(1)
    P = rng.uniform(0, 1, (data.N, data.T)) * data.p_upper
    s = optimal_slack(data, P)
    num = objective_components(data, P, s)
    cvx = cvx_components(data, cp.Constant(P), cp.Constant(s))
    for k in num:
        assert float(cvx[k].value) == pytest.approx(num[k], rel=1e-9, abs=1e-12)


def test_prox_peak_is_minimiser():
    data = random_instance(10, 2, seed=3)
    w, rho = Weights(5, 1, 1), 0.3
    v = np.random.default_rng(2).normal(30, 40, data.T)
    a = peak_quadratic_coeff(data, w)
    f = lambda L: a * np.sum(L ** 2) + 0.5 * rho * np.sum((L - v) ** 2)
    L_hat = prox_peak(data, w, v, rho)
    assert np.all(L_hat >= 0) and np.all(L_hat <= data.grid_cap + 1e-12)
    rng = np.random.default_rng(3)
    for _ in range(200):
        L = np.clip(L_hat + rng.normal(0, 1, data.T), 0, data.grid_cap)
        assert f(L) >= f(L_hat) - 1e-9