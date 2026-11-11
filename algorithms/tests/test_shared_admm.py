import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from algorithms.shared_admm.shared_admm import run_shared_admm
from centralized.solver import solve_centralized
from common.constraints import constraint_violations
from common.objective import WEIGHT_CASES
from ev.fleet import scenario_instance

GAP_TOL = 1e-2          # 1 % of J*
VIOL_TOL = 0.5          # kW on station / grid limits


@pytest.fixture(scope="module")
def inst():
    data, _ = scenario_instance("morning_peak", 20, 2, seed=0)
    stars = {n: solve_centralized(data, w).objective for n, w in WEIGHT_CASES.items()}
    return data, stars


@pytest.mark.parametrize("adaptive", [False, True])
def test_converges_to_centralized(inst, adaptive):
    data, stars = inst
    w = WEIGHT_CASES["baseline"]
    r = run_shared_admm(data, w, adaptive=adaptive, max_iter=5000, J_star=stars["baseline"])
    gap = abs(r.objective - stars["baseline"]) / abs(stars["baseline"])
    print(f"\nadaptive={adaptive}: iters={r.iterations} converged={r.converged} "
          f"gap={gap:.2e} viol={r.metrics['max_violation']:.3f} t={r.runtime_s:.2f}s")
    assert r.converged
    assert gap < GAP_TOL
    assert r.metrics["max_violation"] < VIOL_TOL


@pytest.mark.parametrize("case", list(WEIGHT_CASES))
def test_all_weight_cases(inst, case):
    data, stars = inst
    r = run_shared_admm(data, WEIGHT_CASES[case], adaptive=True, max_iter=5000)
    gap = abs(r.objective - stars[case]) / abs(stars[case])
    print(f"\n{case}: iters={r.iterations} gap={gap:.2e}")
    assert gap < GAP_TOL


def test_local_constraints_always_hold(inst):
    data, _ = inst
    r = run_shared_admm(data, WEIGHT_CASES["baseline"], max_iter=50)
    v = constraint_violations(data, r.P, r.s)
    for key in ("power_lower", "power_upper", "soc_max", "energy_req", "slack_bounds"):
        assert v[key] < 1e-6, key


def test_star_message_count(inst):
    data, _ = inst
    r = run_shared_admm(data, WEIGHT_CASES["baseline"], max_iter=30, eps_abs=0.0, eps_rel=0.0)
    assert r.iterations == 30
    assert r.messages == 2 * data.N * 30
    assert r.floats_sent == 2 * data.N * data.T * 30


def test_residuals_decrease(inst):
    data, _ = inst
    r = run_shared_admm(data, WEIGHT_CASES["baseline"], max_iter=2000)
    rp = r.history["r_primal"]
    assert rp[-1] < rp[0]