"""
Objective  J = J_cost + alpha J_peak + beta J_delay + gamma J_fair

    J_cost  = 1/C_ref * sum_t pi_t dt L_t
    J_peak  = 1/T * sum_t (L_t / P_ref)^2,           L_t = sum_i p_{i,t}
J_delay = sum_i sum_t dt/H_ref * (R_i - e_{i,t}) / R_i.
    J_fair  = sum_i (s_i/R_i) + (s_i / R_i)^2

Every term is provided in two forms:
  * NumPy  -> evaluate any schedule (centralized, ADMM iterate, uncoordinated)
  * CVXPY  -> build the centralized problem / local ADMM subproblems

Separable structure used by ADMM
--------------------------------
J_cost, J_delay are linear in P and J_fair is per-EV, so they split across EV
agents:  f_i(p_i, s_i) = c_i^T p_i + gamma (s_i / R_i)^2  with c = linear_coefficients().
Only J_peak couples the agents through the aggregate L; prox_peak() solves its
proximal step in closed form.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np

from common.utils import ProblemData


# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Weights:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0

    def __post_init__(self):
        if min(self.alpha, self.beta, self.gamma) < 0:
            raise ValueError("weights must be non-negative")

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


# Experiment 2 cases (same weights for centralized and distributed runs)
WEIGHT_CASES: Dict[str, Weights] = {
    "baseline":          Weights(1.0, 1.0, 1.0),
    "peak_priority":     Weights(10.0, 1.0, 1.0),
    "delay_priority":    Weights(1.0, 10.0, 1.0),
    "fairness_priority": Weights(1.0, 1.0, 10.0),
}

COMPONENTS = ("J_cost", "J_peak", "J_delay", "J_fair")


# ======================================================================
# NumPy evaluation
# ======================================================================
def aggregate_load(P: np.ndarray) -> np.ndarray:
    """L_t = sum_i p_{i,t}  -> (T,)"""
    return np.asarray(P).sum(axis=0)


def j_cost(data: ProblemData, P: np.ndarray) -> float:
    return float(data.prices @ aggregate_load(P) * data.dt / data.C_ref)


def j_peak(data: ProblemData, P: np.ndarray) -> float:
    L = aggregate_load(P)
    return float(np.mean((L / data.P_ref) ** 2))


def j_delay(data: ProblemData, P: np.ndarray) -> float:
    return float(data.delay_const + np.sum(data.delay_weight * P))

def j_fair(data: ProblemData, s: np.ndarray) -> float:
    x = np.asarray(s) * data.inv_R
    return float(np.sum(x + x ** 2))


def optimal_slack(data: ProblemData, P: np.ndarray) -> np.ndarray:
    """Smallest feasible s for a given P:  s_i = max(0, R_i - eta_i dt sum_t p_{i,t}).
    This is also the optimal s for fixed P, since J_fair increases with s."""
    delivered = data.eta * data.dt * np.asarray(P).sum(axis=1)
    return np.maximum(data.R - delivered, 0.0)


def objective_components(data: ProblemData, P: np.ndarray, s: np.ndarray | None = None) -> Dict[str, float]:
    if s is None:
        s = optimal_slack(data, P)
    return {"J_cost": j_cost(data, P), "J_peak": j_peak(data, P),
            "J_delay": j_delay(data, P), "J_fair": j_fair(data, s)}


def total_objective(data: ProblemData, P: np.ndarray, s: np.ndarray | None,
                    w: Weights) -> float:
    c = objective_components(data, P, s)
    return c["J_cost"] + w.alpha * c["J_peak"] + w.beta * c["J_delay"] + w.gamma * c["J_fair"]


# ======================================================================
# Helpers for distributed algorithms
# ======================================================================
def linear_coefficients(data: ProblemData, w: Weights) -> np.ndarray:
    """(N, T) matrix c with  J_cost + beta J_delay = sum_{i,t} c[i,t] p[i,t]."""
    cost = (data.prices * data.dt / data.C_ref)[None, :]
    return np.broadcast_to(cost, (data.N, data.T)) + w.beta * data.delay_weight


def peak_quadratic_coeff(data: ProblemData, w: Weights) -> float:
    """alpha J_peak = a * ||L||^2  with  a = alpha / (T P_ref^2)."""
    return w.alpha / (data.T * data.P_ref ** 2)


def prox_peak(data: ProblemData, w: Weights, v: np.ndarray, rho: float,
              enforce_grid_cap: bool = True) -> np.ndarray:
    """argmin_L  a ||L||^2 + rho/2 ||L - v||^2   s.t. 0 <= L_t <= P_grid,t^max.
    Separable in t, so clipping the unconstrained minimiser is exact."""
    a = peak_quadratic_coeff(data, w)
    L = rho * np.asarray(v) / (2.0 * a + rho)
    if enforce_grid_cap:
        L = np.clip(L, 0.0, data.grid_cap)
    return L


# ======================================================================
# CVXPY expressions
# ======================================================================
def cvx_components(data: ProblemData, P, s) -> Dict[str, object]:
    """CVXPY expressions for each component.  P: Variable (N,T), s: Variable (N,)."""
    import cvxpy as cp

    L = cp.sum(P, axis=0)
    return {
        "J_cost": (data.prices * data.dt / data.C_ref) @ L,
        "J_peak": cp.sum_squares(L / data.P_ref) / data.T,
        "J_delay": data.delay_const + cp.sum(cp.multiply(data.delay_weight, P)),
        "J_fair": cp.sum(cp.multiply(data.inv_R, s)) + cp.sum_squares(cp.multiply(data.inv_R, s)),
    }


def cvx_objective(data: ProblemData, P, s, w: Weights):
    c = cvx_components(data, P, s)
    return c["J_cost"] + w.alpha * c["J_peak"] + w.beta * c["J_delay"] + w.gamma * c["J_fair"]
