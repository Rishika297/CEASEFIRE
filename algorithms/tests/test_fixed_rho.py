import numpy as np
import pytest

from algorithms.base import ADMMSettings, auto_rho
from algorithms.rho_update import FixedRho
from algorithms.shared_admm.shared_admm import run_shared_admm
from algorithms.shared_admm.shared_utils import (prox_station_grid, sharing_residuals,
                                                 station_mean)
from common.objective import WEIGHT_CASES
from ev.fleet import scenario_instance


@pytest.fixture(scope="module")
def data():
    return scenario_instance("morning_peak", 20, 3, seed=1)[0]


def test_prox_station_grid_matches_cvxpy(data):
    cp = pytest.importorskip("cvxpy")
    a, rho, N = 0.01, 1.0, data.N
    W = np.random.default_rng(0).normal(1.5, 2.0, (data.M, data.T))   # forces caps to bind
    Z = prox_station_grid(data, a, W, rho)
    Y = N * Z

    Yv = cp.Variable((data.M, data.T))
    f = a * cp.sum_squares(cp.sum(Yv, axis=0)) + rho / (2 * N) * cp.sum_squares(Yv - N * W)
    cons = [Yv >= 0, Yv <= data.station_cap[:, None], cp.sum(Yv, axis=0) <= data.grid_cap]
    cp.Problem(cp.Minimize(f), cons).solve()

    ours = a * np.sum(Y.sum(0) ** 2) + rho / (2 * N) * np.sum((Y - N * W) ** 2)
    assert np.all(Y >= -1e-9) and np.all(Y <= data.station_cap[:, None] + 1e-9)
    assert np.all(Y.sum(0) <= data.grid_cap + 1e-9)
    assert ours == pytest.approx(f.value, rel=1e-6, abs=1e-8)
    assert np.allclose(Y, Yv.value, atol=1e-3)


def test_residuals_zero_at_fixed_point(data):
    P = np.random.default_rng(1).uniform(0, 1, (data.N, data.T)) * data.p_upper
    xbar = station_mean(data, P)
    r, s, *_ = sharing_residuals(data, P, P, xbar, xbar, xbar, xbar, np.zeros_like(xbar), 1.0)
    assert r == pytest.approx(0, abs=1e-12) and s == pytest.approx(0, abs=1e-12)


def test_fixed_rule_never_changes():
    rule = FixedRho()
    assert rule.update(0, 0.5, 1e6, 1e-6) == 0.5


def test_rho_constant_in_fixed_run(data):
    r = run_shared_admm(data, WEIGHT_CASES["baseline"], adaptive=False, max_iter=200)
    assert len(set(r.history["rho"])) == 1
    assert auto_rho(data, WEIGHT_CASES["baseline"]) > 0


def test_settings_file_loads():
    st = ADMMSettings.load()
    assert st.max_iter > 0 and isinstance(st.eps_abs, float)