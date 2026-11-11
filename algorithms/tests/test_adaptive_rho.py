import numpy as np
import pytest

from algorithms.base import auto_rho
from algorithms.rho_update import ResidualBalancing, rescale_scaled_dual
from algorithms.shared_admm.shared_admm import run_shared_admm
from common.objective import WEIGHT_CASES
from ev.fleet import scenario_instance


def test_residual_balancing_rules():
    rb = ResidualBalancing(mu=10, tau_incr=2, tau_decr=2, adapt_until=5)
    assert rb.update(0, 1.0, r=100.0, s=1.0) == 2.0      # primal too big -> increase
    assert rb.update(1, 1.0, r=1.0, s=100.0) == 0.5      # dual too big   -> decrease
    assert rb.update(2, 1.0, r=1.0, s=2.0) == 1.0        # balanced       -> keep
    assert rb.update(5, 1.0, r=100.0, s=1.0) == 1.0      # frozen after adapt_until


def test_rescale_keeps_unscaled_dual():
    u = np.array([1.0, -2.0, 3.0])
    assert np.allclose(2.0 * u, 8.0 * rescale_scaled_dual(u, 2.0, 8.0))


def test_adaptive_recovers_from_bad_rho():
    cp = pytest.importorskip("cvxpy")
    from centralized.solver import solve_centralized
    data, _ = scenario_instance("morning_peak", 20, 2, seed=0)
    w = WEIGHT_CASES["baseline"]
    J_star = solve_centralized(data, w).objective
    bad = 100 * auto_rho(data, w)
    fixed = run_shared_admm(data, w, adaptive=False, rho=bad, max_iter=3000)
    adapt = run_shared_admm(data, w, adaptive=True, rho=bad, max_iter=3000)
    print(f"\nbad rho start: fixed iters={fixed.iterations} gap={abs(fixed.objective-J_star)/J_star:.2e} | "
          f"adaptive iters={adapt.iterations} gap={abs(adapt.objective-J_star)/J_star:.2e} "
          f"rho {adapt.meta['rho_initial']:.2e}->{adapt.meta['rho_final']:.2e}")
    assert adapt.meta["n_rho_changes"] > 0
    assert abs(adapt.objective - J_star) / abs(J_star) < 1e-2