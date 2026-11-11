"""
End-to-end check: fleet -> ProblemData -> centralized QP -> metrics -> Result.
Instance: morning_peak, 30 EVs, 3 stations, seed 0 (everyone arrives together,
so station/grid limits are binding).
"""
import numpy as np
import pytest

cp = pytest.importorskip("cvxpy")

from centralized.solver import build_problem, solve_problem
from common.constraints import constraint_violations, is_feasible
from common.objective import WEIGHT_CASES, objective_components, optimal_slack, total_objective
from common.result import Result
from ev.fleet import scenario_instance
from evaluation.metrics import compute_metrics


@pytest.fixture(scope="module")
def inst():
    return scenario_instance("morning_peak", n_evs=30, n_stations=3, seed=0)


@pytest.fixture(scope="module")
def sols(inst):
    data, _ = inst
    return {name: solve_problem(build_problem(data, w)) for name, w in WEIGHT_CASES.items()}


def test_cvxpy_objective_equals_numpy(inst, sols):
    data, _ = inst
    for name, sol in sols.items():
        J = total_objective(data, sol.P, sol.s, WEIGHT_CASES[name])
        assert J == pytest.approx(sol.objective, rel=1e-5, abs=1e-6), name


def test_centralized_is_feasible(inst, sols):
    data, _ = inst
    for name, sol in sols.items():
        assert is_feasible(data, sol.P, sol.s, tol=1e-4), (name, constraint_violations(data, sol.P, sol.s))


def test_slack_equals_unmet_energy(inst, sols):
    data, _ = inst
    for sol in sols.values():
        assert np.allclose(sol.s, optimal_slack(data, sol.P), atol=1e-3)


def test_asap_breaks_limits_centralized_lowers_peak(inst, sols):
    data, evs = inst
    P_asap = np.vstack([ev.asap_schedule(data.T, data.dt) for ev in evs])
    v = constraint_violations(data, P_asap)
    assert v["station_cap"] > 0 or v["grid_cap"] > 0
    assert sols["baseline"].P.sum(0).max() < P_asap.sum(0).max()


@pytest.mark.parametrize("case,term", [("peak_priority", "J_peak"),
                                       ("delay_priority", "J_delay"),
                                       ("fairness_priority", "J_fair")])
def test_raising_a_weight_never_increases_its_term(sols, case, term):
    # theory: if only weight k grows, J_k at the optimum cannot grow
    base, new = sols["baseline"].components[term], sols[case].components[term]
    assert new <= base + 1e-4 * max(1.0, abs(base))


def test_metrics_sane(inst, sols):
    data, _ = inst
    m = compute_metrics(data, sols["baseline"].P, sols["baseline"].s, WEIGHT_CASES["baseline"])
    assert m["max_violation"] <= 1e-4
    assert 0 <= m["target_soc_pct"] <= 100
    assert 0 < m["jain_fairness"] <= 1 + 1e-12
    assert m["J_total"] == pytest.approx(sols["baseline"].objective, rel=1e-5, abs=1e-6)


def test_result_save_load(inst, sols, tmp_path):
    data, _ = inst
    sol = sols["baseline"]
    r = Result("centralized", weights=WEIGHT_CASES["baseline"].as_dict(), scenario="morning_peak")
    r.finalize(sol.P, sol.s, sol.objective, objective_components(data, sol.P, sol.s))
    r2 = Result.load(r.save(tmp_path / "run"))
    assert np.allclose(r2.P, sol.P) and r2.objective == pytest.approx(sol.objective)


def test_print_comparison(inst, sols):
    """Not a check: prints the table you will put in the report (run with -s)."""
    data, evs = inst
    P_asap = np.vstack([ev.asap_schedule(data.T, data.dt) for ev in evs])
    keys = ("peak_load_kw", "total_cost_inr", "avg_delay_h", "target_soc_pct", "jain_fairness", "max_violation")
    print("\nuncoordinated", {k: round(compute_metrics(data, P_asap)[k], 2) for k in keys})
    for name, sol in sols.items():
        print(f"{name:14s}", {k: round(compute_metrics(data, sol.P, sol.s)[k], 2) for k in keys})