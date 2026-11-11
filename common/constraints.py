"""
Constraints of the EV charging problem.

From the write-up
  (C1) 0 <= p_{i,t} <= P_i^max A_{i,t}                      power + availability
  (C2) SOC_{i,t+1} = SOC_{i,t} + eta_i p_{i,t} dt / B_i       battery dynamics
  (C3) SOC_i^min <= SOC_{i,t} <= SOC_i^max                    battery limits
  (C4) sum_{i in E_m} p_{i,t} <= P_m^max                      station capacity
  (C5) sum_i p_{i,t} <= P_grid,t^max                          shared grid capacity
  (C6) eta_i dt sum_t p_{i,t} + s_i >= R_i                    energy requirement (soft)
  (C7) s_i >= 0

Added
  (C8) s_i <= R_i      -- redundant at the optimum, but bounds the slack box
                          for ADMM and fixes s_i = 0 when R_i = 0.

Implementation notes
  * C2 is eliminated: SOC is an affine function of P,
        SOC_{i,t} = SOC_i^init + eta_i dt / B_i * sum_{tau < t} p_{i,tau},
    so no SOC variables are needed.
  * Because p >= 0 (no V2G), SOC is non-decreasing, so C3's lower bound holds
    whenever SOC_init >= SOC_min (checked in ProblemData) and C3 reduces to the
    final-SOC upper bound
        eta_i dt sum_t p_{i,t} <= B_i (SOC_i^max - SOC_i^init).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from common.utils import ProblemData

TOL = 1e-6


# ======================================================================
# NumPy helpers
# ======================================================================
def delivered_energy(data: ProblemData, P: np.ndarray) -> np.ndarray:
    """Battery-side energy delivered to each EV [kWh] -> (N,)."""
    return data.eta * data.dt * np.asarray(P).sum(axis=1)


def soc_trajectory(data: ProblemData, P: np.ndarray) -> np.ndarray:
    """SOC_{i,t} for t = 0..T  -> (N, T+1)."""
    inc = (data.eta * data.dt / data.battery)[:, None] * np.asarray(P)
    return data.soc_init[:, None] + np.concatenate(
        [np.zeros((data.N, 1)), np.cumsum(inc, axis=1)], axis=1)


def project_local_box(data: ProblemData, P: np.ndarray) -> np.ndarray:
    """Euclidean projection onto C1 (box + availability). Handy for ADMM/heuristics."""
    return np.clip(P, 0.0, data.p_upper)


def constraint_violations(data: ProblemData, P: np.ndarray, s: np.ndarray | None = None) -> Dict[str, float]:
    """Largest violation of each constraint (0 = satisfied). Units: kW or kWh.
    Use this on ADMM iterates, which are only feasible in the limit."""
    P = np.asarray(P)
    L = P.sum(axis=0)
    e = delivered_energy(data, P)
    if s is None:
        s = np.maximum(data.R - e, 0.0)
    s = np.asarray(s)
    return {
        "power_lower": float(np.max(np.maximum(-P, 0.0), initial=0.0)),
        "power_upper": float(np.max(np.maximum(P - data.p_upper, 0.0), initial=0.0)),
        "soc_max": float(np.max(np.maximum(e - data.energy_headroom, 0.0), initial=0.0)),
        "station_cap": float(np.max(np.maximum(data.station_matrix @ P - data.station_cap[:, None], 0.0), initial=0.0)),
        "grid_cap": float(np.max(np.maximum(L - data.grid_cap, 0.0), initial=0.0)),
        "energy_req": float(np.max(np.maximum(data.R - e - s, 0.0), initial=0.0)),
        "slack_bounds": float(np.max(np.maximum(np.maximum(-s, s - data.R), 0.0), initial=0.0)),
    }


def is_feasible(data: ProblemData, P: np.ndarray, s: np.ndarray | None = None, tol: float = 1e-4) -> bool:
    return max(constraint_violations(data, P, s).values()) <= tol


# ======================================================================
# CVXPY constraints
# ======================================================================
def local_constraints(data: ProblemData, P, s, idx=None) -> Dict[str, List]:
    """Per-EV constraints C1, C3, C6, C7, C8 (no coupling).
    These are what each EV agent enforces in its own ADMM subproblem.
    ``idx`` lets the caller pass a single EV (P of shape (1,T) or (T,))."""
    import cvxpy as cp

    if idx is None:
        idx = np.arange(data.N)
    if len(P.shape) == 1:                      # single EV: P is (T,), s scalar
        i = int(np.atleast_1d(idx)[0])
        p_up, R, head = data.p_upper[i], data.R[i], data.energy_headroom[i]
        e = data.eta[i] * data.dt * cp.sum(P)
    else:                                      # several EVs: P is (n, T), s is (n,)
        idx = np.atleast_1d(idx)
        p_up, R, head = data.p_upper[idx], data.R[idx], data.energy_headroom[idx]
        e = cp.multiply(data.eta[idx] * data.dt, cp.sum(P, axis=1))
    return {
        "power_lower": [P >= 0],
        "power_upper": [P <= p_up],
        "soc_max": [e <= head],
        "energy_req": [e + s >= R],
        "slack_lower": [s >= 0],
        "slack_upper": [s <= R],
    }


def coupling_constraints(data: ProblemData, P) -> Dict[str, List]:
    """Constraints that couple EVs: C4 (per station) and C5 (grid).
    Their dual variables are the station / grid congestion prices."""
    import cvxpy as cp

    return {
        "station_cap": [data.station_matrix @ P <= data.station_cap[:, None]],
        "grid_cap": [cp.sum(P, axis=0) <= data.grid_cap],
    }


def build_constraints(data: ProblemData, P, s,
                      include_station: bool = True, include_grid: bool = True) -> Dict[str, List]:
    cons = local_constraints(data, P, s)
    coup = coupling_constraints(data, P)
    if include_station:
        cons["station_cap"] = coup["station_cap"]
    if include_grid:
        cons["grid_cap"] = coup["grid_cap"]
    return cons


def flatten(cons: Dict[str, List]) -> List:
    return [c for group in cons.values() for c in group]
