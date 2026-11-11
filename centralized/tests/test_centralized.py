import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from centralized.solver import solve_centralized
from centralized.uncoordinated import fcfs_schedule, run_uncoordinated
from common.constraints import constraint_violations, is_feasible
from common.objective import WEIGHT_CASES, optimal_slack, total_objective
from common.utils import random_instance
from ev.fleet import scenario_instance


@pytest.fixture(scope="module")
def data():
    return scenario_instance("morning_peak", n_evs=30, n_stations=3, seed=0)[0]


@pytest.fixture(scope="module")
def results(data):
    return {name: solve_centralized(data, w) for name, w in WEIGHT_CASES.items()}


def test_objective_matches_numpy(data, results):
    for name, r in results.items():
        assert total_objective(data, r.P, r.s, WEIGHT_CASES[name]) == pytest.approx(r.objective, rel=1e-5, abs=1e-6)


def test_feasible(data, results):
    for name, r in results.items():
        assert is_feasible(data, r.P, r.s, tol=1e-4), (name, constraint_violations(data, r.P, r.s))
        assert np.allclose(r.s, optimal_slack(data, r.P), atol=1e-3)


def test_better_than_feasible_baseline(data, results):
    # FCFS is feasible, so the optimum can only be as good or better
    w = WEIGHT_CASES["baseline"]
    P = fcfs_schedule(data)
    assert results["baseline"].objective <= total_objective(data, P, optimal_slack(data, P), w) + 1e-6


def test_lower_peak_than_asap(data, results):
    asap = run_uncoordinated(data, WEIGHT_CASES["baseline"], "asap")
    assert results["baseline"].metrics["peak_load_kw"] < asap.metrics["peak_load_kw"]


@pytest.mark.parametrize("case,term", [("peak_priority", "J_peak"),
                                       ("delay_priority", "J_delay"),
                                       ("fairness_priority", "J_fair")])
def test_raising_weight_never_increases_its_term(results, case, term):
    base, new = results["baseline"].components[term], results[case].components[term]
    assert new <= base + 1e-4 * max(1.0, abs(base))


def test_result_fields(results):
    r = results["baseline"]
    assert r.algorithm == "centralized" and r.converged and r.runtime_s > 0
    assert r.metrics["max_violation"] <= 1e-4
    assert len(r.meta["grid_price_dual"]) > 0


def test_small_random_instance():
    d = random_instance(N=20, M=2, seed=0)
    r = solve_centralized(d, WEIGHT_CASES["baseline"])
    assert is_feasible(d, r.P, r.s, tol=1e-4)