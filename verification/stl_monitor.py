"""
STL monitor: checks a charging schedule against Requirement 1 (target SoC
before departure) and Requirement 2 (grid capacity) and returns a report with
Boolean verdicts and robustness values.

Works on any (N, T) schedule, so uncoordinated, centralized and every ADMM
variant are verified identically.

Usage
    from verification import verify_schedule, verify_result
    rep = verify_schedule(data, P, algorithm="centralized")
    print(rep.text())
    rep.to_row()            # flat dict of stl_* columns for a results table
    rep.ev_table()          # per-EV pandas DataFrame

    rep = verify_result(res, data)   # also writes stl_* into res.metrics

CLI
    python -m verification.stl_monitor --algorithm centralized --scenario morning_peak
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from common.utils import ProblemData
from verification.stl_specifications import (GRID_SIGNAL, Specifications,
                                             build_specifications, build_trace,
                                             soc_signal)

# Numerical tolerances for the Boolean verdict (robustness itself is reported raw).
# ADMM iterates are only feasible in the limit, so a tiny negative margin is
# treated as "satisfied at solver precision".
TOL_SOC = 1e-3        # 0.1 percentage point of SoC
TOL_GRID = 1e-3       # 0.1 % of grid capacity


# ======================================================================
# Result containers
# ======================================================================
@dataclass
class EVVerdict:
    ev_id: int
    station: int
    arrival: int                    # step
    departure: int                  # step (exclusive)
    arrival_clock: float            # hour of day
    departure_clock: float
    soc_init: float
    soc_target: float
    soc_at_departure: float
    rho: float                      # robustness of F_[a,d](SoC >= target), SoC units
    satisfied: bool
    t_reached: Optional[int]        # first step the target is met (None = never)
    reached_clock: Optional[float]
    deadline_margin_h: Optional[float]  # time between reaching target and departure
                                        # (temporal robustness of the F operator)
    reachable: bool                 # could the EV reach the target charging alone at P_max?

    @property
    def status(self) -> str:
        if self.satisfied:
            return "SATISFIED"
        return "VIOLATED" if self.reachable else "VIOLATED (unreachable)"


@dataclass
class GridViolation:
    start: int                      # first step over capacity
    end: int                        # last step over capacity (inclusive)
    start_clock: float
    duration_h: float
    worst_excess_kw: float
    overload_kwh: float             # energy above the limit during the episode
    top_contributors: List[Dict]    # [{ev_id, kw}] at the worst step


@dataclass
class GridVerdict:
    cap_kw: float                   # max capacity over the horizon
    peak_kw: float
    t_peak: int
    rho_kw: float                   # min_t (P_grid,t - L_t)
    rho_norm: float                 # min_t (P_grid,t - L_t) / P_grid,t
    t_critical: int                 # step with the smallest margin
    satisfied: bool
    violations: List[GridViolation] = field(default_factory=list)

    @property
    def n_violation_steps(self) -> int:
        return int(sum(v.end - v.start + 1 for v in self.violations))


@dataclass
class STLReport:
    algorithm: str
    scenario: Optional[str]
    dt: float
    start_hour: float
    evs: List[EVVerdict]
    grid: GridVerdict
    rho_req1: float                 # min_i rho_i  (robustness of AND_i phi_i)
    rho_overall: float              # min(rho_req1, grid rho_norm)
    margin_kw: np.ndarray = field(repr=False, default=None)   # P_grid - L_t, (T,)

    # ---------------- aggregate verdicts ----------------
    @property
    def req1_satisfied(self) -> bool:
        return all(e.satisfied for e in self.evs)

    @property
    def req2_satisfied(self) -> bool:
        return self.grid.satisfied

    @property
    def satisfied(self) -> bool:
        return self.req1_satisfied and self.req2_satisfied

    @property
    def violated_evs(self) -> List[EVVerdict]:
        return [e for e in self.evs if not e.satisfied]

    # ---------------- exports ----------------
    def to_row(self) -> Dict[str, float]:
        n = max(len(self.evs), 1)
        viol = self.violated_evs
        reach = [e for e in self.evs if e.reachable]
        margins = [e.deadline_margin_h for e in self.evs
                   if e.satisfied and e.deadline_margin_h is not None]
        return {
            "stl_satisfied": self.satisfied,
            "stl_rho_overall": self.rho_overall,
            "stl_req1_satisfied": self.req1_satisfied,
            "stl_req1_rho_min": self.rho_req1,
            "stl_req1_rho_mean": float(np.mean([e.rho for e in self.evs])) if self.evs else 0.0,
            "stl_req1_sat_pct": 100.0 * (len(self.evs) - len(viol)) / n,
            "stl_req1_n_violated": len(viol),
            "stl_req1_n_unreachable": sum(not e.reachable for e in viol),
            # same check restricted to EVs whose target is physically reachable:
            # separates algorithm failures from impossible requests
            "stl_req1_reachable_sat_pct": 100.0 * sum(e.satisfied for e in reach) / max(len(reach), 1),
            "stl_req1_reachable_n_violated": sum(not e.satisfied for e in reach),
            "stl_satisfied_reachable": all(e.satisfied for e in reach) and self.req2_satisfied,
            # temporal robustness: how long before departure the target was reached
            "stl_req1_min_margin_h": min(margins) if margins else float("nan"),
            "stl_req1_mean_margin_h": float(np.mean(margins)) if margins else float("nan"),
            "stl_req2_satisfied": self.req2_satisfied,
            "stl_req2_rho_kw": self.grid.rho_kw,
            "stl_req2_rho_norm": self.grid.rho_norm,
            "stl_req2_viol_steps": self.grid.n_violation_steps,
            "stl_req2_viol_hours": self.grid.n_violation_steps * self.dt,
            "stl_req2_overload_kwh": float(sum(v.overload_kwh for v in self.grid.violations)),
        }

    def ev_table(self):
        import pandas as pd
        rows = []
        for e in self.evs:
            d = asdict(e)
            d["status"] = e.status
            rows.append(d)
        return pd.DataFrame(rows)

    def text(self, max_evs: Optional[int] = 15) -> str:
        from verification.verification_report import format_report
        return format_report(self, max_evs=max_evs)


# ======================================================================
# Monitor
# ======================================================================
def _clock(step, dt: float, start_hour: float) -> float:
    return float((start_hour + step * dt) % 24.0)


def _runs(mask: np.ndarray):
    """Contiguous True runs in a boolean array -> list of (start, end_inclusive)."""
    out, t, T = [], 0, mask.size
    while t < T:
        if mask[t]:
            s = t
            while t + 1 < T and mask[t + 1]:
                t += 1
            out.append((s, t))
        t += 1
    return out


def check_soc_targets(data: ProblemData, trace, specs: Specifications,
                      tol: float = TOL_SOC) -> List[EVVerdict]:
    """Requirement 1 / deadline monitoring for every EV."""
    out = []
    for i, phi in enumerate(specs.soc):
        soc = trace[soc_signal(i)]
        a, d = int(data.arrival[i]), int(data.departure[i])
        target = float(data.soc_target[i])
        rho = phi.rho(trace)
        hit = np.flatnonzero(soc[a:d + 1] >= target - tol)
        t_hit = int(a + hit[0]) if hit.size else None
        reachable = float(data.eta[i] * data.p_max[i] * (d - a) * data.dt) \
            >= float(data.R[i]) - 1e-9 and target <= data.soc_max[i] + 1e-12
        out.append(EVVerdict(
            ev_id=int(data.ev_ids[i]), station=int(data.station[i]),
            arrival=a, departure=d,
            arrival_clock=_clock(a, data.dt, data.start_hour),
            departure_clock=_clock(d, data.dt, data.start_hour),
            soc_init=float(data.soc_init[i]), soc_target=target,
            soc_at_departure=float(soc[d]), rho=float(rho),
            satisfied=bool(rho >= -tol), t_reached=t_hit,
            reached_clock=None if t_hit is None else _clock(t_hit, data.dt, data.start_hour),
            deadline_margin_h=None if t_hit is None else (d - t_hit) * data.dt,
            reachable=bool(reachable),
        ))
    return out


def check_grid_capacity(data: ProblemData, P: np.ndarray, trace, specs: Specifications,
                        tol: float = TOL_GRID, n_contributors: int = 3):
    """Requirement 2: margin trajectory, robustness, violation episodes."""
    L = trace[GRID_SIGNAL]
    cap = np.asarray(data.grid_cap, dtype=float)
    margin = cap - L                                       # robustness signal r(t) [kW]
    rho_kw = specs.grid.rho(trace)
    rho_norm = specs.grid_normalized.rho(trace)
    t_crit = int(np.argmin(margin / np.where(cap > 0, cap, 1.0)))
    t_peak = int(np.argmax(L))

    violations = []
    for s, e in _runs(margin < -tol * cap):
        seg = slice(s, e + 1)
        t_worst = s + int(np.argmin(margin[seg]))
        col = P[:, t_worst]
        top = np.argsort(col)[::-1][:n_contributors]
        violations.append(GridViolation(
            start=s, end=e, start_clock=_clock(s, data.dt, data.start_hour),
            duration_h=(e - s + 1) * data.dt,
            worst_excess_kw=float(-margin[t_worst]),
            overload_kwh=float(np.sum(np.maximum(-margin[seg], 0.0)) * data.dt),
            top_contributors=[{"ev_id": int(data.ev_ids[j]), "kw": float(col[j])}
                              for j in top if col[j] > 1e-9],
        ))
    verdict = GridVerdict(
        cap_kw=float(cap.max()), peak_kw=float(L.max()), t_peak=t_peak,
        rho_kw=float(rho_kw), rho_norm=float(rho_norm), t_critical=t_crit,
        satisfied=bool(rho_norm >= -tol), violations=violations,
    )
    return verdict, margin


def verify_schedule(data: ProblemData, P: np.ndarray, algorithm: str = "",
                    scenario: Optional[str] = None, tol_soc: float = TOL_SOC,
                    tol_grid: float = TOL_GRID) -> STLReport:
    """Run the STL monitor on one schedule."""
    P = np.asarray(P, dtype=float)
    specs = build_specifications(data)
    trace = build_trace(data, P)
    evs = check_soc_targets(data, trace, specs, tol_soc)
    grid, margin = check_grid_capacity(data, P, trace, specs, tol_grid)
    rho_req1 = min((e.rho for e in evs), default=np.inf)
    return STLReport(
        algorithm=algorithm, scenario=scenario, dt=data.dt, start_hour=data.start_hour,
        evs=evs, grid=grid, rho_req1=float(rho_req1),
        rho_overall=float(min(rho_req1, grid.rho_norm)), margin_kw=margin,
    )


def verify_result(res, data: ProblemData, **kw) -> STLReport:
    """Verify a common.result.Result and add the stl_* values to res.metrics."""
    rep = verify_schedule(data, res.P, algorithm=res.algorithm, scenario=res.scenario, **kw)
    res.metrics.update(rep.to_row())
    return rep


def stl_metrics(data: ProblemData, P: np.ndarray, **kw) -> Dict[str, float]:
    """Flat stl_* dict, convenient to merge into evaluation.metrics output."""
    return verify_schedule(data, P, **kw).to_row()


# ======================================================================
if __name__ == "__main__":
    import argparse

    from common.objective import WEIGHT_CASES
    from ev.fleet import scenario_instance
    from network.integration import run_algorithm

    p = argparse.ArgumentParser(description="STL verification of one schedule")
    p.add_argument("--algorithm", default="centralized")
    p.add_argument("--topology", default="random")
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-factor", type=float, default=None)
    p.add_argument("--case", default="baseline", choices=list(WEIGHT_CASES))
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--all-evs", action="store_true", help="list every EV in the report")
    a = p.parse_args()

    extra = {} if a.grid_factor is None else {"grid_cap_factor": a.grid_factor}
    data, _ = scenario_instance(a.scenario, a.n_evs, a.stations, a.seed, **extra)
    kw = {} if a.algorithm in ("centralized",) or a.algorithm.startswith("uncoordinated") \
        else {"max_iter": a.max_iter}
    res = run_algorithm(a.algorithm, data, WEIGHT_CASES[a.case], topology=a.topology,
                        scenario=a.scenario, seed=a.seed, **kw)
    print(verify_result(res, data).text(max_evs=None if a.all_evs else 15))
