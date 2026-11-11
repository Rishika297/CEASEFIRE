"""
Building blocks of (proximal) dual consensus ADMM for the EV problem.

Notation (per EV i, all M x T matrices unless stated)
    x_i   power profile (T,), placed in row m(i) by embed()
    w_i   EV i's local estimate of the station loads Y
    E_i v_i = embed(x_i) - w_i / N        (sum_i E_i v_i = 0  <=>  mean_i w_i = Y)
    lambda_i  local copy of the price of the coupling constraint
    p_i       multiplier of the consensus constraints lambda_i = lambda_j
"""
from __future__ import annotations

import numpy as np

from common.utils import ProblemData


def embed(data: ProblemData, X: np.ndarray) -> np.ndarray:
    """(N, T) powers -> (N, M, T) with EV i's profile in row m(i)."""
    out = np.zeros((data.N, data.M, data.T))
    out[np.arange(data.N), data.station] = X
    return out


def own_rows(data: ProblemData, R: np.ndarray) -> np.ndarray:
    """(N, M, T) -> (N, T): row m(i) of agent i's matrix."""
    return R[np.arange(data.N), data.station]


def coupling(data: ProblemData, X: np.ndarray, Wg: np.ndarray) -> np.ndarray:
    """E_i v_i for every agent -> (N, M, T)."""
    return embed(data, X) - Wg / data.N


CONSENSUS_C_FACTOR = 10.0   # calibrated with experiments/compare_fixed_adaptive.py


def auto_c(degrees: np.ndarray, N: int, delta: float = 1.05) -> float:
    """Scaled penalty c: the value that makes the local proximal weight tau_i
    about 1 for an agent of average degree, times a calibrated factor."""
    return CONSENSUS_C_FACTOR * delta * (1.0 + 1.0 / N ** 2) / (2.0 * float(np.mean(degrees)))


def proximal_taus(c: float, degrees: np.ndarray, N: int, delta: float = 1.05):
    """Block proximal weights (tau_x for the powers, tau_w for the load estimates).

    PDC-ADMM needs Gamma_i = diag(tau_x I, tau_w I) - E_i^T E_i / (2 c n_i) to be
    positive semidefinite. Since ||S x - w/N||^2 <= 2||x||^2 + 2||w||^2 / N^2,
    it is enough that
        tau_x > 1 / (c n_i)     and     tau_w > 1 / (c n_i N^2).
    A separate, N^2-smaller weight for w lets the load estimates move as fast as
    the powers (a single weight made them N times slower, so the grid price was
    never learned when the grid limit binds)."""
    tau_x = delta / (c * np.asarray(degrees, dtype=float))
    return tau_x, tau_x / N ** 2


def prox_aggregate_batch(data: ProblemData, a: float, C: np.ndarray, tau: np.ndarray,
                         iters: int = 60) -> np.ndarray:
    """For every agent i (batched):

        w_i = argmin_w (1/N) g(w) + tau_i/2 ||w - C_i||^2
        g(w) = a ||sum_m w_m||^2 + I{0 <= w_m <= P_m^max} + I{sum_m w_m <= P_grid^max}

    Multiplying by N and using KKT, for every agent and time step:
        w_m(theta) = clip(C_m - theta / (N tau_i), 0, P_m^max),  theta = 2 a S + lambda_grid
    solved by bisection on theta (same logic as shared_utils.prox_station_grid).
    C: (B, M, T), tau: (B,)  ->  (B, M, T)."""
    N = data.N
    C = np.asarray(C, dtype=float)
    Ntau = (N * np.asarray(tau, dtype=float))[:, None]          # (B, 1)
    kinv = (1.0 / Ntau)[:, :, None]                              # (B, 1, 1)
    cap = data.station_cap[None, :, None]                        # (1, M, 1)
    G = data.grid_cap[None, :]                                   # (1, T)

    def W_of(theta):                                             # theta (B, T)
        return np.clip(C - kinv * theta[:, None, :], 0.0, cap)

    def S_of(theta):
        return W_of(theta).sum(axis=1)                           # (B, T)

    lo = np.minimum((C - cap).min(axis=1) * Ntau, 0.0) - 2.0 * a * cap.sum() - 1.0
    hi = np.maximum(C.max(axis=1) * Ntau, 0.0) + 1.0
    hi_all = hi.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = mid - 2.0 * a * S_of(mid) > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    theta = hi

    over = S_of(theta) > G + 1e-12
    if np.any(over):
        lo2, hi2 = theta.copy(), hi_all
        for _ in range(iters):
            mid = 0.5 * (lo2 + hi2)
            pos = S_of(mid) > G
            lo2 = np.where(pos, mid, lo2)
            hi2 = np.where(pos, hi2, mid)
        theta = np.where(over, hi2, theta)
    return W_of(theta)


def edge_differences(Lam: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """lambda_i - lambda_j for every edge -> (|E|, M, T)."""
    if edges.size == 0:
        return np.zeros((0,) + Lam.shape[1:])
    return Lam[edges[:, 0]] - Lam[edges[:, 1]]