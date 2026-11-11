from __future__ import annotations

import math


def admm_tolerances(n, x_norm, z_norm, y_norm, eps_abs, eps_rel):
    eps_pri = math.sqrt(n) * eps_abs + eps_rel * max(x_norm, z_norm)
    eps_dual = math.sqrt(n) * eps_abs + eps_rel * y_norm
    return eps_pri, eps_dual


def has_converged(r_primal, r_dual, eps_pri, eps_dual):
    return r_primal <= eps_pri and r_dual <= eps_dual


MAX_VIOLATION_KW = 0.1
