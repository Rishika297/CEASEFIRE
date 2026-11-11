"""Tests for the STL engine, the two specifications and the monitor."""
import numpy as np
import pytest

from centralized.uncoordinated import asap_schedule, fcfs_schedule
from common.utils import ProblemData, random_instance, tou_prices
from verification.stl import Always, And, Eventually, Not, Or, Predicate
from verification.stl_monitor import verify_schedule
from verification.stl_specifications import build_trace, grid_capacity_spec, soc_target_spec


# ----------------------------------------------------------------------
# STL engine
# ----------------------------------------------------------------------
def test_predicate_robustness_grid_example():
    # 100 kW limit: 80 -> +20, 95 -> +5, 100 -> 0, 110 -> -10
    tr = {"P": np.array([80.0, 95.0, 100.0, 110.0])}
    mu = Predicate("P", "<=", 100.0)
    assert [mu.rho(tr, t) for t in range(4)] == [20.0, 5.0, 0.0, -10.0]
    assert Always(mu, 0, 3).rho(tr) == -10.0           # min
    assert Always(mu, 0, 2).rho(tr) == 0.0
    assert Eventually(mu, 0, 3).rho(tr) == 20.0        # max


def test_boolean_operators_and_windows():
    tr = {"x": np.array([0.0, 1.0, 2.0, 3.0])}
    a, b = Predicate("x", ">=", 2.0), Predicate("x", "<=", 1.0)
    assert And(a, b).rho(tr, 0) == min(a.rho(tr, 0), b.rho(tr, 0))
    assert Or(a, b).rho(tr, 0) == max(a.rho(tr, 0), b.rho(tr, 0))
    assert Not(a).rho(tr, 3) == -a.rho(tr, 3)
    assert Eventually(a, 1, 10).rho(tr) == 1.0          # window clipped to signal length
    assert Always(a, 5, 6).rho(tr) == np.inf            # empty window: vacuously true
    assert Eventually(a, 5, 6).rho(tr) == -np.inf
    assert Always(a, 0, 3).critical_time(tr) == 0
    assert Eventually(a, 0, 3).critical_time(tr) == 3
    # nested formula takes the slow (generic) path
    assert Always(Eventually(a, 0, 1), 0, 2).rho(tr) == -1.0   # worst t=0: max(-2, -1)


def test_scaled_predicate_is_dimensionless():
    tr = {"P": np.array([50.0, 120.0])}
    assert Always(Predicate("P", "<=", 100.0, scale=100.0), 0, 1).rho(tr) == pytest.approx(-0.2)


# ----------------------------------------------------------------------
# Specifications on a hand-built 3-EV case (the EV1/EV2/EV3 deadline example)
# ----------------------------------------------------------------------
def _three_ev_data(grid_cap=100.0):
    T, dt = 8, 1.0
    return ProblemData(
        T=T, dt=dt, prices=np.ones(T), grid_cap=grid_cap, station_cap=np.array([1e3]),
        station=np.zeros(3), battery=np.array([50.0, 50.0, 50.0]),
        soc_init=np.array([0.30, 0.30, 0.30]), soc_target=np.array([0.90, 0.80, 0.95]),
        eta=np.ones(3), p_max=np.array([20.0, 20.0, 20.0]),
        arrival=np.array([0, 0, 0]), departure=np.array([4, 5, 6]))


def _schedule_three(data):
    P = np.zeros((3, data.T))
    P[0, :3] = 10.0                 # 30 kWh = +60 pp -> 90 %  (meets 90 %)
    P[1, :3] = [10.0, 10.0, 2.5]    # 22.5 kWh = +45 pp -> 75 % (misses 80 %)
    P[2, 1:5] = [10.0, 10.0, 10.0, 2.5]   # 32.5 kWh -> 95 %       (meets 95 %)
    return P


def test_deadline_example():
    data = _three_ev_data()
    P = _schedule_three(data)
    rep = verify_schedule(data, P)
    status = {e.ev_id: e.satisfied for e in rep.evs}
    assert status == {0: True, 1: False, 2: True}
    assert rep.evs[1].rho == pytest.approx(-0.05)         # 75 % vs 80 % target
    assert rep.evs[1].soc_at_departure == pytest.approx(0.75)
    assert rep.evs[0].t_reached == 3 and rep.evs[0].deadline_margin_h == pytest.approx(1.0)
    assert rep.req2_satisfied and not rep.req1_satisfied and not rep.satisfied
    assert rep.grid.rho_kw == pytest.approx(100.0 - 30.0)
    txt = rep.text()
    assert "Overall STL status: VIOLATED" in txt and "REQUIREMENT 2 - grid capacity: SATISFIED" in txt


def test_charging_after_departure_does_not_count():
    data = _three_ev_data()
    P = np.zeros((3, data.T))
    P[0, 4:7] = 10.0      # all of EV0's charging happens after it left (step >= 4)
    tr = build_trace(data, P)
    assert soc_target_spec(data, 0).rho(tr) == pytest.approx(0.30 - 0.90)


def test_grid_violation_episode_and_contributors():
    data = _three_ev_data(grid_cap=25.0)
    P = _schedule_three(data)       # load 20, 30, 22.5, 10, 2.5, 0 -> only step 1 > 25
    rep = verify_schedule(data, P)
    assert not rep.req2_satisfied
    assert rep.grid.rho_kw == pytest.approx(-5.0)
    assert grid_capacity_spec(data, normalized=True).rho(build_trace(data, P)) == pytest.approx(-0.2)
    assert len(rep.grid.violations) == 1
    v = rep.grid.violations[0]
    assert (v.start, v.end) == (1, 1) and v.overload_kwh == pytest.approx(5.0)
    assert {c["ev_id"] for c in v.top_contributors} <= {0, 1, 2}
    assert rep.rho_overall == pytest.approx(min(rep.rho_req1, -0.2))


def test_unreachable_target_is_flagged():
    data = _three_ev_data()
    data2 = ProblemData(**{**{k: getattr(data, k) for k in (
        "T", "dt", "prices", "grid_cap", "station_cap", "station", "battery", "soc_init",
        "soc_target", "eta", "arrival", "departure")}, "p_max": np.array([20.0, 20.0, 1.0])})
    rep = verify_schedule(data2, np.clip(_schedule_three(data), 0, data2.p_upper))
    ev2 = rep.evs[2]
    assert not ev2.satisfied and not ev2.reachable and ev2.status == "VIOLATED (unreachable)"


# ----------------------------------------------------------------------
# Algorithms
# ----------------------------------------------------------------------
def test_fcfs_respects_grid_and_asap_can_violate():
    data = random_instance(N=20, M=2, seed=3, grid_cap=25.0)
    rep_fcfs = verify_schedule(data, fcfs_schedule(data))
    rep_asap = verify_schedule(data, asap_schedule(data))
    assert rep_fcfs.req2_satisfied
    assert not rep_asap.req2_satisfied and rep_asap.grid.rho_kw < 0


def test_centralized_satisfies_grid_and_feasible_targets():
    pytest.importorskip("cvxpy")
    from centralized.solver import solve_centralized
    from common.objective import WEIGHT_CASES
    data = random_instance(N=12, M=2, seed=1)
    res = solve_centralized(data, WEIGHT_CASES["baseline"])
    rep = verify_schedule(data, res.P)
    assert rep.req2_satisfied
    # every EV that the centralized solution could serve (s_i ~ 0) passes Req 1
    for e, s in zip(rep.evs, res.s):
        if s < 1e-4:
            assert e.satisfied


def test_row_keys():
    data = _three_ev_data()
    row = verify_schedule(data, _schedule_three(data)).to_row()
    assert row["stl_req1_n_violated"] == 1 and row["stl_req1_sat_pct"] == pytest.approx(200 / 3)
    assert row["stl_req2_viol_hours"] == 0
