"""
Centralized convex QP

    min_{P, s}  J_cost + alpha J_peak + beta J_delay + gamma J_fair
    s.t.        C1-C8  (see common/constraints.py)

Always feasible (P = 0, s = R), convex, and the reference optimum J* that every
ADMM variant is compared against.

Usage
    res = solve_centralized(data, WEIGHT_CASES["baseline"])
    res.objective, res.metrics["peak_load_kw"], res.P

Quick demo (from project root):
    python -m centralized.solver
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from common.constraints import build_constraints, flatten
from common.objective import Weights, cvx_components
from common.result import Result
from common.utils import ProblemData, Timer
from evaluation.metrics import compute_metrics


@dataclass
class Formulation:
    data: ProblemData
    weights: Weights
    problem: object                       # cvxpy.Problem
    P: object                             # cvxpy.Variable (N, T)
    s: object                             # cvxpy.Variable (N,)
    constraints: Dict[str, List]
    components: Dict[str, object]


@dataclass
class Solution:
    status: str
    P: np.ndarray                         # (N, T) kW
    s: np.ndarray                         # (N,)   kWh
    objective: float
    components: Dict[str, float]
    duals: Dict[str, np.ndarray] = field(default_factory=dict)
    solve_time: Optional[float] = None


def default_solver() -> Optional[str]:
    import cvxpy as cp
    installed = cp.installed_solvers()
    for name in ("CLARABEL", "ECOS", "OSQP", "SCS"):
        if name in installed:
            return name
    return None


def build_problem(data: ProblemData, weights: Weights,
                  include_station: bool = True, include_grid: bool = True) -> Formulation:
    import cvxpy as cp

    P = cp.Variable((data.N, data.T), name="P")
    s = cp.Variable(data.N, name="s")
    cons = build_constraints(data, P, s, include_station, include_grid)
    comps = cvx_components(data, P, s)
    obj = (comps["J_cost"] + weights.alpha * comps["J_peak"]
           + weights.beta * comps["J_delay"] + weights.gamma * comps["J_fair"])
    prob = cp.Problem(cp.Minimize(obj), flatten(cons))
    return Formulation(data, weights, prob, P, s, cons, comps)


def solve_problem(form: Formulation, solver: Optional[str] = None,
                  verbose: bool = False, **kwargs) -> Solution:
    import cvxpy as cp

    form.problem.solve(solver=solver or default_solver(), verbose=verbose, **kwargs)
    status = form.problem.status
    if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"centralized problem not solved: status={status}")

    # remove tiny numerical noise from the interior-point solver
    P = np.clip(form.P.value, 0.0, form.data.p_upper)
    s = np.clip(form.s.value, 0.0, form.data.R)

    duals = {}
    for name in ("station_cap", "grid_cap"):
        if name in form.constraints:
            duals[name] = np.asarray(form.constraints[name][0].dual_value)

    return Solution(
        status=status, P=P, s=s, objective=float(form.problem.value),
        components={k: float(v.value) for k, v in form.components.items()},
        duals=duals, solve_time=form.problem.solver_stats.solve_time,
    )


def solve_centralized(data: ProblemData, weights: Weights, solver: Optional[str] = None,
                      scenario: Optional[str] = None, seed: Optional[int] = None,
                      verbose: bool = False, **kwargs) -> Result:
    """Build + solve + evaluate. Returns the common Result object."""
    with Timer() as tm:
        form = build_problem(data, weights)
        sol = solve_problem(form, solver, verbose, **kwargs)
    res = Result(algorithm="centralized", weights=weights.as_dict(), scenario=scenario,
                 n_evs=data.N, n_stations=data.M, seed=seed)
    res.finalize(sol.P, sol.s, sol.objective, sol.components, converged=True, iterations=1)
    res.runtime_s = tm.elapsed
    res.metrics = compute_metrics(data, sol.P, sol.s, weights)
    res.meta.update(
        status=sol.status,
        solver=solver or default_solver(),
        solver_time_s=sol.solve_time,
        grid_price_dual=sol.duals.get("grid_cap", np.array([])).tolist(),
        station_price_dual=sol.duals.get("station_cap", np.array([])).tolist(),
    )
    return res


if __name__ == "__main__":
    from common.objective import WEIGHT_CASES
    from centralized.uncoordinated import run_uncoordinated
    from ev.fleet import scenario_instance

    data, _ = scenario_instance("morning_peak", n_evs=30, n_stations=3, seed=0)
    keys = ("peak_load_kw", "total_cost_inr", "avg_delay_h", "target_soc_pct",
            "jain_fairness", "max_violation")
    w0 = WEIGHT_CASES["baseline"]
    for mode in ("asap", "fcfs"):
        r = run_uncoordinated(data, w0, mode)
        print(f"{r.algorithm:22s} J={r.objective:8.4f}", {k: round(r.metrics[k], 2) for k in keys})
    for name, w in WEIGHT_CASES.items():
        r = solve_centralized(data, w)
        print(f"centralized/{name:10s} J={r.objective:8.4f} t={r.runtime_s:.2f}s",
              {k: round(r.metrics[k], 2) for k in keys})