"""
Unified performance metrics.  Works on any schedule P (N, T): uncoordinated,
centralized or an ADMM iterate, so every algorithm is judged identically.

Metric definitions
------------------
peak_load_kw          max_t L_t
peak_to_average       max_t L_t / mean_t L_t      (mean over the whole horizon)
load_factor           mean_t L_t / max_t L_t
load_std_kw           std_t L_t
total_cost_inr        sum_t pi_t dt L_t
energy_grid_kwh       sum_t dt L_t                (grid side)
energy_battery_kwh    sum_i eta_i dt sum_t p_it   (battery side)
avg_delay_h           mean over EVs of the energy-weighted wait (t - a_i) dt
avg_time_to_target_h  mean over EVs that reach target of the time from arrival
                      to the end of the interval where target was reached
target_soc_pct        % EVs with delivered >= (1 - tol) R_i   (EVs with R_i > 0)
mean_satisfaction     mean_i min(delivered_i / R_i, 1)
min_satisfaction      min_i  min(delivered_i / R_i, 1)
jain_fairness         (sum x)^2 / (n sum x^2) on satisfaction ratios x_i
total_unmet_kwh       sum_i s_i
max_violation         largest constraint violation (0 for a feasible schedule)
J_*, J_total          objective components (and total if weights are given)
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from common.constraints import constraint_violations, delivered_energy
from common.objective import Weights, aggregate_load, objective_components, optimal_slack
from common.utils import ProblemData

SAT_TOL = 1e-3 # an EV "reaches target" if it gets >= 99.9% of R_i


def jain_index(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(x == 0):
        return 1.0 if x.size else float("nan")
    return float(x.sum() ** 2 / (x.size * np.sum(x ** 2)))


def satisfaction_ratio(data: ProblemData, P: np.ndarray) -> np.ndarray:
    """min(delivered_i / R_i, 1) for EVs with R_i > 0."""
    need = data.R > 1e-9
    return np.minimum(delivered_energy(data, P)[need] / data.R[need], 1.0)


def average_delay(data: ProblemData, P: np.ndarray) -> float:
    """Energy-weighted mean waiting time per EV [h], averaged over EVs that charge."""
    P = np.asarray(P)
    wait = (np.arange(data.T)[None, :] - data.arrival[:, None]) * data.dt
    e = P.sum(axis=1)
    charged = e > 1e-9
    if not np.any(charged):
        return 0.0
    per_ev = (np.where(data.avail, wait, 0.0) * P).sum(axis=1)[charged] / e[charged]
    return float(per_ev.mean())


def time_to_target(data: ProblemData, P: np.ndarray, tol: float = SAT_TOL) -> np.ndarray:
    """Hours from arrival until R_i is delivered (NaN if never reached / R_i = 0)."""
    cum = np.cumsum((data.eta * data.dt)[:, None] * np.asarray(P), axis=1)
    reached = cum >= (1 - tol) * data.R[:, None]
    out = np.full(data.N, np.nan)
    for i in np.flatnonzero((data.R > 1e-9) & reached.any(axis=1)):
        t_done = int(np.argmax(reached[i]))
        out[i] = (t_done + 1 - data.arrival[i]) * data.dt
    return out


def compute_metrics(data: ProblemData, P: np.ndarray, s: Optional[np.ndarray] = None,
                    weights: Optional[Weights] = None) -> Dict[str, float]:
    P = np.asarray(P)
    if s is None:
        s = optimal_slack(data, P)
    L = aggregate_load(P)
    peak, mean = float(L.max()), float(L.mean())
    sat = satisfaction_ratio(data, P)
    ttt = time_to_target(data, P)
    viol = constraint_violations(data, P, s)

    m = {
        "n_evs": data.N,
        "peak_load_kw": peak,
        "peak_to_average": peak / mean if mean > 0 else float("nan"),
        "load_factor": mean / peak if peak > 0 else float("nan"),
        "load_std_kw": float(L.std()),
        "total_cost_inr": float(data.prices @ L * data.dt),
        "energy_grid_kwh": float(L.sum() * data.dt),
        "energy_battery_kwh": float(delivered_energy(data, P).sum()),
        "avg_delay_h": average_delay(data, P),
        "avg_time_to_target_h": float(np.nanmean(ttt)) if np.any(~np.isnan(ttt)) else float("nan"),
        "target_soc_pct": float(100.0 * np.mean(sat >= 1 - SAT_TOL)) if sat.size else 100.0,
        "mean_satisfaction": float(sat.mean()) if sat.size else 1.0,
        "min_satisfaction": float(sat.min()) if sat.size else 1.0,
        "jain_fairness": jain_index(sat),
        "total_unmet_kwh": float(np.sum(s)),
        "max_violation": max(viol.values()),
    }
    m.update({f"viol_{k}": v for k, v in viol.items()})
    m.update(objective_components(data, P, s))
    if weights is not None:
        m["J_total"] = (m["J_cost"] + weights.alpha * m["J_peak"]
                        + weights.beta * m["J_delay"] + weights.gamma * m["J_fair"])
    return m


def optimality_gap(J: float, J_star: float) -> float:
    """Relative gap |J - J*| / max(|J*|, 1e-12)."""
    return abs(J - J_star) / max(abs(J_star), 1e-12)


def metrics_table(results: Mapping[str, Dict[str, float]]):
    """{'centralized': metrics, 'shared_admm': metrics, ...} -> pandas DataFrame."""
    import pandas as pd
    return pd.DataFrame(results).T