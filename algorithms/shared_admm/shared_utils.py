"""
Building blocks of Shared (sharing) ADMM for the EV problem.

Each EV i owns x_i = its power profile placed in the row of its station inside
an (M, T) matrix (all other rows are zero).  The coupled function is

    g(Y) = alpha/(T P_ref^2) ||sum_m Y_m||^2           (peak term)
         + I{0 <= Y_m <= P_m^max}  for every station m (station capacity)
         + I{sum_m Y_m <= P_grid^max}                  (grid capacity)

where Y = sum_i x_i is the (M, T) matrix of station loads.

Sharing ADMM (Boyd et al. 2011, Sec. 7.3), scaled form:
    x_i  <- argmin f_i(x_i) + rho/2 ||x_i - x_i^k + xbar^k - zbar^k + u^k||^2
    zbar <- argmin g(N zbar) + N rho/2 ||zbar - u^k - xbar^{k+1}||^2
    u    <- u^k + xbar^{k+1} - zbar^{k+1}
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from common.utils import ProblemData


def station_mean(data: ProblemData, P: np.ndarray) -> np.ndarray:
    """xbar = (1/N) sum_i x_i  ->  (M, T) = station loads / N."""
    return data.station_matrix @ np.asarray(P) / data.N


def local_targets(data: ProblemData, P: np.ndarray, xbar: np.ndarray,
                  zbar: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Proximal centre of every EV:  v_i = p_i - (xbar - zbar + u)[m(i)]  ->  (N, T)."""
    return np.asarray(P) - (xbar - zbar + u)[data.station]


def prox_station_grid(data: ProblemData, a: float, W: np.ndarray, rho: float,
                      iters: int = 60) -> np.ndarray:
    """zbar-update.  With Y = N zbar this solves, independently for every t,

        min_Y  a (sum_m Y_m)^2 + rho/(2N) sum_m (Y_m - N W_m)^2
        s.t.   0 <= Y_m <= P_m^max,   sum_m Y_m <= P_grid,t^max

    KKT:  Y_m(theta) = clip(N W_m - (N/rho) theta, 0, P_m^max),
          theta = 2 a S + lambda,  S = sum_m Y_m,  lambda >= 0 (grid multiplier).
    theta - 2a S(theta) is increasing -> bisection; if the root breaks the grid
    limit, a second bisection finds theta with S(theta) = P_grid^max.
    Returns zbar (M, T)."""
    N = data.N
    V = N * np.asarray(W, dtype=float)              # (M, T)
    cap = data.station_cap[:, None]                 # (M, 1)
    G = data.grid_cap                               # (T,)
    k = N / rho

    def Y_of(theta):
        return np.clip(V - k * theta[None, :], 0.0, cap)

    def S_of(theta):
        return Y_of(theta).sum(axis=0)

    # at lo every station is at capacity, at hi every station is at 0
    lo = np.minimum((V - cap).min(axis=0) / k, 0.0) - 2.0 * a * cap.sum() - 1.0
    hi = np.maximum(V.max(axis=0) / k, 0.0) + 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pos = mid - 2.0 * a * S_of(mid) > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    theta = hi

    over = S_of(theta) > G + 1e-12
    if np.any(over):
        lo2 = theta.copy()
        hi2 = np.maximum(V.max(axis=0) / k, 0.0) + 1.0  # S(hi2) = 0 <= G
        for _ in range(iters):
            mid = 0.5 * (lo2 + hi2)
            pos = S_of(mid) > G
            lo2 = np.where(pos, mid, lo2)
            hi2 = np.where(pos, hi2, mid)
        theta = np.where(over, hi2, theta)

    return Y_of(theta) / N


def sharing_residuals(data: ProblemData, P: np.ndarray, P_old: np.ndarray,
                      xbar: np.ndarray, zbar: np.ndarray,
                      xbar_old: np.ndarray, zbar_old: np.ndarray,
                      u: np.ndarray, rho: float) -> Tuple[float, float, float, float, float]:
    """Primal/dual residual norms and the norms used by the stopping rule.

    z_i = x_i - xbar + zbar, so  x_i - z_i = xbar - zbar  for every i:
        ||r|| = sqrt(N) ||xbar - zbar||
        ||s|| = rho sqrt( sum_i ||z_i^{k+1} - z_i^k||^2 )
    Returns (r, s, ||x||, ||z||, ||y||)."""
    N = data.N
    r = float(np.sqrt(N) * np.linalg.norm(xbar - zbar))

    D = (zbar - xbar) - (zbar_old - xbar_old)      # change of (zbar - xbar), (M, T)
    dP = P - P_old
    dz2 = N * np.sum(D ** 2) + 2.0 * np.sum(dP * D[data.station]) + np.sum(dP ** 2)
    s = float(rho * np.sqrt(max(dz2, 0.0)))

    Dz = zbar - xbar
    z2 = N * np.sum(Dz ** 2) + 2.0 * np.sum(P * Dz[data.station]) + np.sum(P ** 2)
    x_norm = float(np.linalg.norm(P))
    z_norm = float(np.sqrt(max(z2, 0.0)))
    y_norm = float(rho * np.sqrt(N) * np.linalg.norm(u))
    return r, s, x_norm, z_norm, y_norm