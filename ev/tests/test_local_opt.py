import numpy as np
import pytest

from common.objective import Weights, linear_coefficients
from ev.fleet import make_instance
from ev.local_opt import local_objective, prox_local, prox_local_cvxpy, selfish_schedule


@pytest.fixture(scope="module")
def setup():
    data, _ = make_instance(15, 2, "evening_peak", "high", seed=4)
    w = Weights(1, 1, 5)
    c = linear_coefficients(data, w)
    v = np.random.default_rng(0).normal(2, 3, (data.N, data.T))
    return data, w, c, v


def test_feasible(setup):
    data, w, c, v = setup
    P, s, _ = prox_local(data, c, v, 0.5, w.gamma)
    E = data.eta * data.dt * P.sum(1)
    assert np.all(P >= -1e-12) and np.all(P <= data.p_upper + 1e-12)
    assert np.all(E <= data.energy_headroom + 1e-8)
    assert np.all(E + s >= data.R - 1e-8) and np.all(s >= 0)


def test_large_rho_returns_projection_of_v(setup):
    data, w, c, v = setup
    P, _, _ = prox_local(data, c, v, 1e6, w.gamma)
    target = np.clip(v, 0, data.p_upper)
    ok = data.eta * data.dt * target.sum(1) <= data.energy_headroom
    assert np.allclose(P[ok], target[ok], atol=1e-3)


def test_matches_cvxpy(setup):
    pytest.importorskip("cvxpy")
    data, w, c, v = setup
    P, s, _ = prox_local(data, c, v, 0.5, w.gamma)
    f = local_objective(data, c, v, 0.5, w.gamma, P, s)
    for i in range(data.N):
        _, _, f_ref = prox_local_cvxpy(data, i, c[i], v[i], 0.5, w.gamma)
        assert f[i] == pytest.approx(f_ref, rel=1e-4, abs=1e-6)


def test_selfish_meets_demand_when_possible(setup):
    data, w, c, _ = setup
    P, s = selfish_schedule(data, c, gamma=1e4)
    feasible_alone = data.eta * data.dt * data.p_upper.sum(1) >= data.R
    assert np.all(s[feasible_alone] < 1e-3 * np.maximum(data.R[feasible_alone], 1))
