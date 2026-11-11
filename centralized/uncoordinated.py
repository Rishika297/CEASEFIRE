"""
Uncoordinated baselines.

asap : every EV charges at full power from arrival until its target
       (or SOC_max) is reached. Respects only its own limits, so station/grid
       capacity CAN be violated. This is the proposal's "uncoordinated" baseline.
fcfs : first-come-first-served. Same greedy behaviour, but the station and
       grid limits are enforced in order of arrival. Always feasible, so its
       objective is an upper bound on the centralized optimum J*.

Usage
    res = run_uncoordinated(data, WEIGHT_CASES["baseline"], mode="asap")
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from common.objective import Weights, objective_components, optimal_slack, total_objective
from common.result import Result
from common.utils import ProblemData, Timer
from evaluation.metrics import compute_metrics


def asap_schedule(data: ProblemData) -> np.ndarray:
    k = data.eta * data.dt
    need = np.minimum(data.R, data.energy_headroom)          # battery-side kWh still needed
    P = np.zeros((data.N, data.T))
    for t in range(data.T):
        p = np.minimum(data.p_upper[:, t], need / k)
        P[:, t] = p
        need = np.maximum(need - k * p, 0.0)
    return P


def fcfs_schedule(data: ProblemData) -> np.ndarray:
    k = data.eta * data.dt
    need = np.minimum(data.R, data.energy_headroom)
    order = np.lexsort((np.arange(data.N), data.arrival))   # earliest arrival first
    P = np.zeros((data.N, data.T))
    for t in range(data.T):
        grid_left = float(data.grid_cap[t])
        st_left = data.station_cap.astype(float).copy()
        for i in order:
            if need[i] <= 1e-12 or not data.avail[i, t]:
                continue
            m = data.station[i]
            p = min(data.p_max[i], need[i] / k[i], st_left[m], grid_left)
            if p <= 0:
                continue
            P[i, t] = p
            need[i] -= k[i] * p
            st_left[m] -= p
            grid_left -= p
    return P


def run_uncoordinated(data: ProblemData, weights: Weights, mode: str = "asap",
                      scenario: Optional[str] = None, seed: Optional[int] = None) -> Result:
    with Timer() as tm:
        if mode == "asap":
            P = asap_schedule(data)
        elif mode == "fcfs":
            P = fcfs_schedule(data)
        else:
            raise ValueError("mode must be 'asap' or 'fcfs'")
    s = optimal_slack(data, P)
    res = Result(algorithm=f"uncoordinated_{mode}", weights=weights.as_dict(),
                 scenario=scenario, n_evs=data.N, n_stations=data.M, seed=seed)
    res.finalize(P, s, total_objective(data, P, s, weights),
                 objective_components(data, P, s), converged=True, iterations=1)
    res.runtime_s = tm.elapsed
    res.metrics = compute_metrics(data, P, s, weights)
    return res