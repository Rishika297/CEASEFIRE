from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FixedRho:
    def update(self, k, rho, r, s):
        return rho


@dataclass
class ResidualBalancing:
    mu: float = 10.0
    tau_incr: float = 2.0
    tau_decr: float = 2.0
    adapt_until: int = 300
    rho_min: float = 1e-12
    rho_max: float = 1e12
    n_changes: int = 0

    def update(self, k, rho, r, s):
        if k >= self.adapt_until:
            return rho
        new = rho
        if r > self.mu * s:
            new = rho * self.tau_incr
        elif s > self.mu * r:
            new = rho / self.tau_decr
        new = float(np.clip(new, self.rho_min, self.rho_max))
        if new != rho:
            self.n_changes += 1
        return new


def rescale_scaled_dual(u, rho_old, rho_new):
    return u * (rho_old / rho_new)


def make_rho_rule(adaptive, mu=10.0, tau_incr=2.0, tau_decr=2.0, adapt_until=300):
    if adaptive:
        return ResidualBalancing(mu=mu, tau_incr=tau_incr, tau_decr=tau_decr, adapt_until=adapt_until)
    return FixedRho()
