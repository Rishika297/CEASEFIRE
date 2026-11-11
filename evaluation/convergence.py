"""
Convergence analysis helpers on common.result.Result histories.

    iters_to_gap(res, 1e-3)    first iteration with objective gap <= 0.1% and violation <= viol_tol
    iters_to_dist(res, 1e-2)   first iteration with load-profile distance ||L - L*|| / ||L*|| <= 1%
                               and violation <= viol_tol  (better at large N, where J* is
                               dominated by the unavoidable unmet-energy penalty)
    messages_to_gap / messages_to_dist   messages sent until then
    convergence_summary(res)   dict of all of the above + final residuals
"""
from __future__ import annotations

from typing import Dict

import numpy as np

VIOL_TOL_KW = 0.1


def _first_index(res, key: str, eps: float, viol_tol: float):
    h = res.history
    if key not in h:
        return None
    for idx, (g, v) in enumerate(zip(h[key], h["max_violation"])):
        if g <= eps and v <= viol_tol:
            return idx
    return None


def _iters(res, key, eps, viol_tol) -> float:
    idx = _first_index(res, key, eps, viol_tol)
    return np.nan if idx is None else float(res.history["iteration"][idx] + 1)


def _msgs(res, key, eps, viol_tol) -> float:
    idx = _first_index(res, key, eps, viol_tol)
    return np.nan if idx is None else float(res.history["messages"][idx])


def iters_to_gap(res, eps: float, viol_tol: float = VIOL_TOL_KW) -> float:
    return _iters(res, "gap", eps, viol_tol)


def iters_to_dist(res, eps: float, viol_tol: float = VIOL_TOL_KW) -> float:
    return _iters(res, "load_dist", eps, viol_tol)


def messages_to_gap(res, eps: float, viol_tol: float = VIOL_TOL_KW) -> float:
    return _msgs(res, "gap", eps, viol_tol)


def messages_to_dist(res, eps: float, viol_tol: float = VIOL_TOL_KW) -> float:
    return _msgs(res, "load_dist", eps, viol_tol)


def convergence_summary(res) -> Dict[str, float]:
    h = res.history
    out = {
        "it_gap_1e-3": iters_to_gap(res, 1e-3),
        "it_gap_1e-4": iters_to_gap(res, 1e-4),
        "it_dist_1e-2": iters_to_dist(res, 1e-2),
        "it_dist_1e-3": iters_to_dist(res, 1e-3),
        "msgs_gap_1e-3": messages_to_gap(res, 1e-3),
        "msgs_dist_1e-2": messages_to_dist(res, 1e-2),
        "final_load_dist": h["load_dist"][-1] if "load_dist" in h else np.nan,
        "final_r_primal": h["r_primal"][-1] if "r_primal" in h else np.nan,
        "final_r_dual": h["r_dual"][-1] if "r_dual" in h else np.nan,
        "MB_total": res.floats_sent * 8 / 1e6,
    }
    if "avg_rounds" in h:
        out["mean_avg_rounds"] = float(np.mean(h["avg_rounds"]))
    return out