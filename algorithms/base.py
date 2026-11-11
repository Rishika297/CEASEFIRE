"""
Shared pieces for every ADMM variant: settings, automatic rho, load distance,
message counter.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import Optional, Union

import numpy as np

from common.objective import Weights, linear_coefficients, peak_quadratic_coeff
from common.utils import ProblemData, load_yaml


@dataclass
class ADMMSettings:
    rho: Union[float, str] = "auto"
    max_iter: int = 3000
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    adaptive_rho: bool = False
    mu: float = 3.0                     # residual-balancing threshold for Shared ADMM
    mu_consensus: float = 10.0          # residual-balancing threshold for Consensus ADMM
    tau_incr: float = 2.0
    tau_decr: float = 2.0
    adapt_until: int = 300
    log_every: int = 1
    verbose: bool = False
    J_star: Optional[float] = None        # centralized optimum, if known (logs the gap)
    L_star: Optional[object] = None       # centralized load profile (T,), if known (logs load_dist)
    store_iterates: bool = False          # keep P every store_every iterations (animations)
    store_every: int = 10

    @classmethod
    def from_dict(cls, d: dict) -> "ADMMSettings":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown ADMM settings: {sorted(unknown)}")
        return cls(**d)

    @classmethod
    def load(cls, path: str = "configs/admm_config.yaml", **overrides) -> "ADMMSettings":
        d = load_yaml(path).get("admm", {})
        d.update(overrides)
        return cls.from_dict(d)

    def with_(self, **overrides) -> "ADMMSettings":
        return replace(self, **overrides)


def auto_rho(data: ProblemData, weights: Weights) -> float:
    """Problem-scaled reference rho: the larger of
       * the curvature the peak term puts on each EV (2 a N), and
       * the size of the linear cost per kW divided by a typical charging power.
    Also used by both ADMM variants for internal scaling (kappa = 1 / auto_rho),
    so do NOT change this function to tune rho; change AUTO_RHO_FACTOR instead."""
    c = linear_coefficients(data, weights)
    a = peak_quadratic_coeff(data, weights)
    c_scale = float(np.mean(np.abs(c[data.avail]))) if data.avail.any() else 0.0
    return max(2.0 * a * data.N, c_scale / float(np.mean(data.p_max)), 1e-8)


AUTO_RHO_FACTOR = 0.3   # calibrated with experiments/compare_fixed_adaptive.py (Shared ADMM)


def resolve_rho(settings: ADMMSettings, data: ProblemData, weights: Weights) -> float:
    """Starting rho for Shared ADMM in original units."""
    if isinstance(settings.rho, str):
        if settings.rho != "auto":
            raise ValueError("rho must be a number or 'auto'")
        return AUTO_RHO_FACTOR * auto_rho(data, weights)
    return float(settings.rho)


def load_distance(P: np.ndarray, L_star) -> float:
    """||L - L*|| / ||L*|| for the total load profile of schedule P."""
    Ls = np.asarray(L_star, dtype=float)
    return float(np.linalg.norm(np.asarray(P).sum(axis=0) - Ls) / max(np.linalg.norm(Ls), 1e-12))


class CommCounter:
    """Counts messages and transmitted scalars (communication overhead)."""

    def __init__(self):
        self.messages = 0
        self.floats = 0

    def add(self, messages: int, floats: int) -> None:
        self.messages += int(messages)
        self.floats += int(floats)