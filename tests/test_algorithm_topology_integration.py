import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from algorithms.shared_admm.shared_admm import run_shared_admm
from centralized.solver import solve_centralized
from common.objective import WEIGHT_CASES
from evaluation.convergence import convergence_summary, iters_to_gap
from ev.fleet import scenario_instance
from network.averaging import GossipAverager, HubAverager
from network.integration import ADMM_ALGORITHMS, make_network, run_algorithm

W = WEIGHT_CASES["baseline"]


@pytest.fixture(scope="module")
def inst():
    data, _ = scenario_instance("morning_peak", 12, 2, seed=0)
    return data, solve_centralized(data, W).objective


def test_hub_averager_equals_coordinator(inst):
    data, J = inst
    a = run_shared_admm(data, W, max_iter=400, J_star=J)
    b = run_shared_admm(data, W, max_iter=400, J_star=J,
                        averager=HubAverager(make_network("star", data.N)), topology="star")
    assert a.iterations == b.iterations
    assert b.objective == pytest.approx(a.objective, rel=1e-6)
    assert b.messages == 2 * (data.N - 1) * b.iterations


def test_mesh_gossip_is_exact(inst):
    data, J = inst
    a = run_shared_admm(data, W, max_iter=400, J_star=J)
    b = run_shared_admm(data, W, max_iter=400, J_star=J,
                        averager=GossipAverager(make_network("mesh", data.N)), topology="mesh")
    assert b.objective == pytest.approx(a.objective, rel=1e-5)
    assert np.all(np.array(b.history["avg_rounds"]) == 1)


def test_shared_on_ring_converges(inst):
    data, J = inst
    r = run_shared_admm(data, W, max_iter=1500, J_star=J,
                        averager=GossipAverager(make_network("ring", data.N), tol=1e-8, max_rounds=5000),
                        topology="ring")
    gap = abs(r.objective - J) / J
    print(f"\nshared on ring: iters={r.iterations} gap={gap:.2e} "
          f"rounds/iter={np.mean(r.history['avg_rounds']):.0f}")
    assert gap < 1e-2


@pytest.mark.parametrize("algo", ADMM_ALGORITHMS)
@pytest.mark.parametrize("topo", ["ring", "star", "mesh", "random"])
def test_dispatch_all_combinations(inst, algo, topo):
    data, J = inst
    r = run_algorithm(algo, data, W, topology=topo, max_iter=5, J_star=J)
    assert r.algorithm == algo and r.topology == topo
    assert r.iterations == 5 and r.messages > 0
    s = convergence_summary(r)
    assert "it_gap_1e-3" in s and "MB_total" in s


def test_iters_to_gap(inst):
    data, J = inst
    r = run_shared_admm(data, W, max_iter=2000, J_star=J)
    k = iters_to_gap(r, 1e-3)
    assert 1 <= k <= r.iterations