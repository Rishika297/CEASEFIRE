"""
STL specifications of the EV charging problem.

Signals built from a schedule P (N, T)  [see build_trace]
    SoC_i[t]    t = 0..T   state of charge at the start of interval t
                           (SoC_i[T] is the final SoC; same convention as
                           common.constraints.soc_trajectory)
    P_total[t]  t = 0..T-1 aggregate charging power  L_t = sum_i p_{i,t}  [kW]

Requirement 1 - every EV reaches its target SoC before it departs
    phi_i^SoC  =  F_[a_i, d_i] ( SoC_i >= SoC_i^target )
    The EV is plugged in during [a_i, d_i) (departure exclusive), so SoC_i[d_i]
    is the charge it leaves with: the window [a_i, d_i] is exactly
    "between arrival and departure".  Robustness is in SoC units
    (0.05 = 5 percentage points above/below target).

Requirement 2 - total charging power never exceeds grid capacity
    phi^grid   =  G_[0, T-1] ( P_total <= P_grid )
    Robustness in kW = min_t (P_grid,t - L_t)  (worst-case headroom).

Overall specification
    phi  =  ( AND_i phi_i^SoC )  AND  phi^grid
    STL conjunction takes the min of the robustness values, which only makes
    sense in common units, so for the overall formula the grid predicate is
    normalised by capacity (P_grid - L)/P_grid -> both terms are fractions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from common.constraints import soc_trajectory
from common.utils import ProblemData
from verification.stl import Always, And, Eventually, Formula, Predicate, Trace


def soc_signal(i: int) -> str:
    return f"SoC_{i}"


GRID_SIGNAL = "P_total"


# ----------------------------------------------------------------------
def build_trace(data: ProblemData, P: np.ndarray) -> Trace:
    """Turn a charging schedule into the signals the specifications read."""
    P = np.asarray(P, dtype=float)
    if P.shape != (data.N, data.T):
        raise ValueError(f"P has shape {P.shape}, expected {(data.N, data.T)}")
    soc = soc_trajectory(data, P)                    # (N, T+1)
    trace: Trace = {soc_signal(i): soc[i] for i in range(data.N)}
    trace[GRID_SIGNAL] = P.sum(axis=0)
    return trace


# ----------------------------------------------------------------------
def soc_target_spec(data: ProblemData, i: int) -> Eventually:
    """Requirement 1 for EV i:  F_[a_i, d_i] (SoC_i >= SoC_i^target)."""
    a, d = int(data.arrival[i]), int(data.departure[i])
    target = float(data.soc_target[i])
    mu = Predicate(soc_signal(i), ">=", target,
                   label=f"(SoC_{i} >= {target:.3f})")
    return Eventually(mu, a, d)


def grid_capacity_spec(data: ProblemData, normalized: bool = False) -> Always:
    """Requirement 2:  G_[0, T-1] (P_total <= P_grid).
    normalized=True divides the margin by the capacity (dimensionless)."""
    cap = np.asarray(data.grid_cap, dtype=float)
    cap_c = float(cap[0]) if np.allclose(cap, cap[0]) else cap
    scale = cap_c if normalized else 1.0
    lbl = (f"(P_total <= {cap_c:.1f} kW)" if np.ndim(cap_c) == 0
           else "(P_total <= P_grid(t))")
    mu = Predicate(GRID_SIGNAL, "<=", cap_c, scale=scale, label=lbl)
    return Always(mu, 0, data.T - 1)


@dataclass
class Specifications:
    soc: List[Eventually]          # Requirement 1, one formula per EV
    grid: Always                   # Requirement 2 (kW)
    grid_normalized: Always        # Requirement 2 (fraction of capacity)
    overall: Formula               # AND of everything (common units)

    def describe(self) -> str:
        lines = [
            "Requirement 1 (per EV i):  F_[a_i, d_i] (SoC_i >= SoC_i^target)",
            "Requirement 2 (grid):      G_[0, T-1]  (P_total <= P_grid)",
            "Overall:                   (AND_i Req1_i) AND Req2",
        ]
        if self.soc:
            lines.append(f"  e.g. EV0:  {self.soc[0]}")
        lines.append(f"  grid:      {self.grid}")
        return "\n".join(lines)


def build_specifications(data: ProblemData) -> Specifications:
    soc = [soc_target_spec(data, i) for i in range(data.N)]
    grid = grid_capacity_spec(data, normalized=False)
    grid_n = grid_capacity_spec(data, normalized=True)
    return Specifications(soc=soc, grid=grid, grid_normalized=grid_n,
                          overall=And(*soc, grid_n))
