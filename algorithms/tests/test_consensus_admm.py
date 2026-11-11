import numpy as np
import pytest

from algorithms.consensus_admm.consensus_admm import run_consensus_admm
from algorithms.consensus_admm.consensus_utils import (coupling, embed, own_rows,
                                                       prox_aggregate_batch, proximal_taus)
from common.constraints import constraint_violations
from common.objective import WEIGHT_CASES
from ev.fleet import scenario_instance
from network.integration import make_network

GAP_TOL = 1e-2
VIOL_TOL = 0.5


# ---------------------------------------------------------------- building blocks
def test_embed_and_rows():
    data, _ = scenario_instance("random", 8, 3, seed=0)
    X = np.random.default_rng(0).uniform(0, 5, (data.N, data.T))
    Y = embed(data, X)
    assert np.allclose(own_rows(data, Y), X)
    assert np.allclose(Y.sum(axis=1), X)                    # other rows are zero
    assert np.allclose(coupling(data, X, np.zeros_like(Y)).sum(0), data.station_matrix @ X)


def test_prox_aggregate_matches_cvxpy():
    cp = pytest.importorskip("cvxpy")
    data, _ = scenario_instance("morning_peak", 20, 3, seed=1)
    a, N = 0.01, data.N
    rng = np.random.default_rng(2)
    C = rng.normal(30, 25, (3, data.M, data.T))             # 3 agents, caps will bind
    tau = np.array([0.5, 1.0, 3.0])
    W = prox_aggregate_batch(data, a, C, tau)
    for i in range(3):
        w = cp.Variable((data.M, data.T))
        f = (a / N) * cp.sum_squares(cp.sum(w, axis=0)) + tau[i] / 2 * cp.sum_squares(w - C[i])
        cons = [w >= 0, w <= data.station_cap[:, None], cp.sum(w, axis=0) <= data.grid_cap]
        cp.Problem(cp.Minimize(f), cons).solve()
        ours = (a / N) * np.sum(W[i].sum(0) ** 2) + tau[i] / 2 * np.sum((W[i] - C[i]) ** 2)
        assert ours == pytest.approx(f.value, rel=1e-6, abs=1e-8)
        assert np.allclose(W[i], w.value, atol=1e-3)


def test_tau_condition():
    deg, N, c = np.array([1, 2, 5, 19]), 20, 0.3
    tau_x, tau_w = proximal_taus(c, deg, N, delta=1.05)
    assert np.all(tau_x * c * deg > 1) and np.all(tau_w * c * deg * N ** 2 > 1)


# ---------------------------------------------------------------- algorithm
@pytest.fixture(scope="module")
def inst():
    pytest.importorskip("cvxpy")
    from centralized.solver import solve_centralized
    data, _ = scenario_instance("morning_peak", 20, 2, seed=0)
    small, _ = scenario_instance("morning_peak", 10, 2, seed=0)
    w = WEIGHT_CASES["baseline"]
    return data, solve_centralized(data, w).objective, small, solve_centralized(small, w).objective


def test_message_count(inst):
    data, *_ = inst
    g = make_network("random", data.N, seed=0)
    r = run_consensus_admm(data, WEIGHT_CASES["baseline"], graph=g, max_iter=25,
                           eps_abs=0.0, eps_rel=0.0)
    assert r.iterations == 25
    assert r.messages == 25 * 2 * g.n_edges
    assert r.floats_sent == r.messages * data.M * data.T


def test_local_constraints_always_hold(inst):
    data, *_ = inst
    r = run_consensus_admm(data, WEIGHT_CASES["baseline"], topology="ring", max_iter=40)
    v = constraint_violations(data, r.P, r.s)
    for key in ("power_lower", "power_upper", "soc_max", "energy_req", "slack_bounds"):
        assert v[key] < 1e-6, key


@pytest.mark.parametrize("topology", ["mesh", "random", "star"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_converges_to_centralized(inst, topology, adaptive):
    data, J_star, *_ = inst
    r = run_consensus_admm(data, WEIGHT_CASES["baseline"], topology=topology,
                           adaptive=adaptive, max_iter=5000, J_star=J_star)
    gap = abs(r.objective - J_star) / abs(J_star)
    print(f"\n{topology:6s} adaptive={adaptive!s:5s} iters={r.iterations} conv={r.converged} "
          f"gap={gap:.2e} viol={r.metrics['max_violation']:.3f} t={r.runtime_s:.1f}s")
    assert gap < GAP_TOL
    assert r.metrics["max_violation"] < VIOL_TOL


@pytest.mark.parametrize("adaptive", [False, True])
def test_ring_small(inst, adaptive):
    *_, small, J_small = inst
    r = run_consensus_admm(small, WEIGHT_CASES["baseline"], topology="ring",
                           adaptive=adaptive, max_iter=10000, J_star=J_small)
    gap = abs(r.objective - J_small) / abs(J_small)
    print(f"\nring(10) adaptive={adaptive!s:5s} iters={r.iterations} conv={r.converged} "
          f"gap={gap:.2e} viol={r.metrics['max_violation']:.3f} t={r.runtime_s:.1f}s")
    assert gap < GAP_TOL