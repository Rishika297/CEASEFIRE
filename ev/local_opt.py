"""
Individual EV local optimisation.

Every ADMM variant needs each EV agent i to solve a proximal problem

    min_{p_i, s_i}  c_i^T p_i + gamma * [ s_i/R_i + (s_i/R_i)^2 ] + (rho_i / 2) ||p_i - v_i||^2
    s.t.            0 <= p_i <= P_i^max A_i                          (C1)
                    eta_i dt sum_t p_it <= B_i (SOC_max - SOC_init)  (C3)
                    eta_i dt sum_t p_it + s_i >= R_i                 (C6)
                    0 <= s_i <= R_i                                  (C7, C8)

where c_i comes from common.objective.linear_coefficients (cost + beta*delay)
and v_i is whatever the ADMM variant passes in (e.g. p_i^k - mean + u^k for
Shared ADMM, or the neighbour average minus the dual for Consensus ADMM).

Fairness penalty (Option A): gamma * [ s/R + (s/R)^2 ]
  * the linear part makes the FIRST unmet kWh cost gamma/R_i, so the EV is
    charged fully whenever the limits allow it;
  * the quadratic part spreads any unavoidable shortfall fairly.

prox_local()        exact solver, vectorised over all EVs (fast, NumPy only)
prox_local_cvxpy()  reference solver for one EV (tests / debugging)
selfish_schedule()  each EV minimises its own cost alone (no coordination)

How prox_local works
--------------------
For fixed energy E_i = eta dt sum p_i the best slack is s_i = max(R_i - E_i, 0),
so the problem only depends on p_i.  KKT conditions give

    p_it(mu) = clip( v_it - (c_it - mu eta dt) / rho_i , 0, P^max A_it )

with a scalar "energy price" mu_i that must satisfy  mu_i = h_i(E_i(mu_i)),

    h_i(E) = gamma/R_i + 2 gamma (R_i - E)/R_i^2    if E < R_i
           = 0                                       if E >= R_i

(the marginal value of one more kWh; when E < R_i it is at least gamma/R_i, the
slope of the linear part, and it drops to 0 once the target is met).
E_i(mu) is non-decreasing and h_i is non-increasing, so mu - h_i(E_i(mu)) is
strictly increasing and the root is found by bisection (it may sit exactly at
E_i = R_i).  If that root over-fills the battery, a second bisection finds the
mu giving E_i = headroom exactly.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from common.utils import ProblemData


def prox_local(data: ProblemData, c: np.ndarray, v: np.ndarray, rho: float | np.ndarray,
               gamma: float, idx: Optional[np.ndarray] = None,
               iters: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the local proximal problem for the EVs in ``idx`` (default: all).

    Parameters
    ----------
    c     : (n, T) linear cost coefficients  (rows for idx)
    v     : (n, T) proximal centre
    rho   : scalar or (n,) penalty, must be > 0
    gamma : fairness weight

    Returns
    -------
    P  : (n, T) optimal powers
    s  : (n,)   optimal unmet energy
    mu : (n,)   energy multiplier (marginal value of 1 kWh for each EV)
    """
    idx = np.arange(data.N) if idx is None else np.atleast_1d(idx)
    n = idx.size
    c = np.broadcast_to(np.asarray(c, dtype=float), (n, data.T))
    v = np.broadcast_to(np.asarray(v, dtype=float), (n, data.T))
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (n,)).copy()
    if np.any(rho <= 0):
        raise ValueError("rho must be > 0 (use selfish_schedule for rho = 0)")
    rho_c = rho[:, None]

    ub = data.p_upper[idx]
    k = data.eta[idx] * data.dt                    # (n,)  energy per kW per step
    R = data.R[idx]
    H = data.energy_headroom[idx]
    lin = gamma * data.inv_R[idx]                  # slope of the linear part   gamma/R
    q = 2.0 * gamma * data.inv_R[idx] ** 2         # slope of the quadratic part 2 gamma/R^2

    def h(E):                                      # marginal value of one more kWh
        gap = R - E
        return np.where(gap > 0, lin + q * gap, 0.0)

    def p_of(mu):
        return np.clip(v - (c - mu[:, None] * k[:, None]) / rho_c, 0.0, ub)

    def E_of(mu):
        return k * p_of(mu).sum(axis=1)

    # bracket: at mu = -big every p is 0, at mu = +big every p hits ub
    big = (np.abs(c).max(axis=1) + rho * (np.abs(v).max(axis=1) + ub.max(axis=1))) / k + 1.0
    lo = -big
    hi = np.maximum(big, lin + q * R) + 1.0        # hi - h(E) > 0 for every E

    # 1) root of mu - h(E(mu)) = 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = mid - h(E_of(mid)) > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    mu = hi

    # 2) battery headroom (SOC_max) active -> choose mu with E(mu) = H
    over = E_of(mu) > H + 1e-12
    if np.any(over):
        lo2, hi2 = -big.copy(), mu.copy()
        for _ in range(iters):
            mid = 0.5 * (lo2 + hi2)
            pos = E_of(mid) > H
            hi2 = np.where(pos, mid, hi2)
            lo2 = np.where(pos, lo2, mid)
        mu = np.where(over, lo2, mu)               # lo2 side always satisfies E <= H

    P = p_of(mu)
    E = k * P.sum(axis=1)
    s = np.clip(R - E, 0.0, R)
    return P, s, mu


def local_objective(data: ProblemData, c: np.ndarray, v: np.ndarray, rho, gamma: float,
                    P: np.ndarray, s: np.ndarray, idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Value of the local proximal objective for each EV -> (n,)."""
    idx = np.arange(data.N) if idx is None else np.atleast_1d(idx)
    rho = np.broadcast_to(np.asarray(rho, dtype=float), (idx.size,))
    x = s * data.inv_R[idx]                        # relative unmet energy s/R
    return (np.sum(c * P, axis=1) + gamma * (x + x ** 2)
            + 0.5 * rho * np.sum((P - v) ** 2, axis=1))


def prox_local_cvxpy(data: ProblemData, i: int, c_i: np.ndarray, v_i: np.ndarray,
                     rho: float, gamma: float, solver: Optional[str] = None):
    """Reference implementation for a single EV (slow, needs cvxpy)."""
    import cvxpy as cp
    from common.constraints import local_constraints

    p = cp.Variable(data.T)
    s = cp.Variable()
    cons = [c_ for group in local_constraints(data, p, s, idx=i).values() for c_ in group]
    x = s * data.inv_R[i]
    obj = (c_i @ p + gamma * (x + cp.square(x))
           + 0.5 * rho * cp.sum_squares(p - v_i))
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=solver)
    return np.asarray(p.value), float(s.value), float(prob.value)


def selfish_schedule(data: ProblemData, c: np.ndarray, gamma: float,
                     rho_reg: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Each EV minimises only its own cost + delay + unmet-energy penalty,
    ignoring station / grid limits (a price-aware but uncoordinated baseline).
    A tiny proximal term (rho_reg, centre 0) makes the LP solution unique."""
    P, s, _ = prox_local(data, c, np.zeros((data.N, data.T)), rho_reg, gamma)
    return P, s
