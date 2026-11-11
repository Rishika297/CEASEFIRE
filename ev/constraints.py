"""
Per-EV (local) constraints, written for a single EV object.

    (C1) 0 <= p_t <= P^max A_t
    (C3) SOC_min <= SOC_t <= SOC_max   ->  eta dt sum_t p_t <= B (SOC_max - SOC_init)
    (C6) eta dt sum_t p_t + s >= R
    (C7) s >= 0
    (C8) s <= R                         (added, see common/constraints.py)

The fleet-level versions (arrays over all EVs, plus station/grid coupling)
live in common/constraints.py; both use exactly the same formulas.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ev.ev_model import EV


def ev_bounds(ev: EV, T: int) -> Tuple[np.ndarray, np.ndarray]:
    """Lower / upper power bounds for each interval (C1)."""
    return np.zeros(T), ev.power_upper(T)


def unmet_energy(ev: EV, p: np.ndarray, dt: float) -> float:
    """Smallest feasible slack s = max(R - delivered, 0)."""
    return max(ev.required_energy_kwh - ev.delivered_energy_kwh(p, dt), 0.0)


def project_box_energy(v: np.ndarray, ub: np.ndarray, e_max: float, iters: int = 100) -> np.ndarray:
    """Euclidean projection of v onto {0 <= p <= ub, sum(p) <= e_max}.
    Solution is p = clip(v - tau, 0, ub) with the smallest tau >= 0 that
    satisfies the sum constraint (found by bisection)."""
    p = np.clip(v, 0.0, ub)
    if p.sum() <= e_max + 1e-12:
        return p
    lo, hi = 0.0, float(np.max(v)) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if np.clip(v - mid, 0.0, ub).sum() > e_max:
            lo = mid
        else:
            hi = mid
    return np.clip(v - hi, 0.0, ub)


def project_power(ev: EV, p: np.ndarray, T: int, dt: float) -> np.ndarray:
    """Project a power profile onto C1 and the SOC_max constraint."""
    e_max = ev.headroom_kwh / (ev.eta * dt)          # in units of sum(p)
    return project_box_energy(np.asarray(p, dtype=float), ev.power_upper(T), e_max)


def check_ev_schedule(ev: EV, p: np.ndarray, dt: float,
                      s: Optional[float] = None) -> Dict[str, float]:
    """Largest violation of each local constraint (0 = satisfied)."""
    p = np.asarray(p, dtype=float)
    T = p.size
    soc = ev.soc_trajectory(p, dt)
    if s is None:
        s = unmet_energy(ev, p, dt)
    R = ev.required_energy_kwh
    return {
        "power_lower": float(max(-p.min(initial=0.0), 0.0)),
        "power_upper": float(np.max(np.maximum(p - ev.power_upper(T), 0.0), initial=0.0)),
        "soc_max": float(max(soc.max() - ev.soc_max, 0.0)),
        "soc_min": float(max(ev.soc_min - soc.min(), 0.0)),
        "energy_req": float(max(R - ev.delivered_energy_kwh(p, dt) - s, 0.0)),
        "slack_bounds": float(max(-s, s - R, 0.0)),
    }


def is_ev_feasible(ev: EV, p: np.ndarray, dt: float, s: Optional[float] = None,
                   tol: float = 1e-6) -> bool:
    return max(check_ev_schedule(ev, p, dt, s).values()) <= tol


def cvx_ev_constraints(ev: EV, p, s, T: int, dt: float) -> List:
    """CVXPY constraints for one EV: p is Variable(T), s is scalar Variable."""
    import cvxpy as cp
    e = ev.eta * dt * cp.sum(p)
    R = ev.required_energy_kwh
    return [p >= 0, p <= ev.power_upper(T), e <= ev.headroom_kwh,
            e + s >= R, s >= 0, s <= R]
