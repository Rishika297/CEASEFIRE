"""
Grid and station capacity model.

The capacity constraints themselves are in common/constraints.py
(coupling_constraints); this module creates capacity variants and analyses
how a schedule uses the capacity.

Usage
    tight = with_capacity(data, grid_factor=0.5)        # 50 % of current grid cap
    util  = utilization(data, P)
"""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

import numpy as np

from common.utils import ProblemData


def with_capacity(data: ProblemData,
                  grid_cap: Optional[float | np.ndarray] = None,
                  station_cap: Optional[np.ndarray] = None,
                  grid_factor: Optional[float] = None,
                  station_factor: Optional[float] = None) -> ProblemData:
    """Copy of ``data`` with new capacities (absolute values or scale factors).
    P_ref and C_ref are kept so objectives stay comparable across variants."""
    gc = np.array(data.grid_cap, dtype=float)
    sc = np.array(data.station_cap, dtype=float)
    if grid_cap is not None:
        gc = np.broadcast_to(np.asarray(grid_cap, dtype=float), (data.T,)).copy()
    if grid_factor is not None:
        gc = gc * grid_factor
    if station_cap is not None:
        sc = np.asarray(station_cap, dtype=float)
    if station_factor is not None:
        sc = sc * station_factor
    if sc.size != data.M:
        raise ValueError("station_cap must have length M")
    return replace(data, grid_cap=gc, station_cap=sc)


def station_loads(data: ProblemData, P: np.ndarray) -> np.ndarray:
    """Load of every station at every interval -> (M, T)."""
    return data.station_matrix @ np.asarray(P)


def congested_intervals(data: ProblemData, P: np.ndarray, tol: float = 1e-3) -> Dict[str, np.ndarray]:
    """Boolean masks of intervals where a capacity limit is (almost) reached."""
    L = np.asarray(P).sum(axis=0)
    S = station_loads(data, P)
    return {
        "grid": L >= data.grid_cap - tol,
        "station": S >= data.station_cap[:, None] - tol,       # (M, T)
    }


def utilization(data: ProblemData, P: np.ndarray) -> Dict[str, float]:
    L = np.asarray(P).sum(axis=0)
    S = station_loads(data, P)
    grid_u = np.divide(L, data.grid_cap, out=np.zeros_like(L), where=data.grid_cap > 0)
    st_u = np.divide(S, data.station_cap[:, None], out=np.zeros_like(S),
                     where=data.station_cap[:, None] > 0)
    cong = congested_intervals(data, P)
    active = L > 1e-9
    return {
        "grid_util_max": float(grid_u.max()),
        "grid_util_mean_active": float(grid_u[active].mean()) if active.any() else 0.0,
        "station_util_max": float(st_u.max()),
        "n_grid_congested": int(cong["grid"].sum()),
        "n_station_congested": int(cong["station"].any(axis=0).sum()),
    }


def capacity_summary(data: ProblemData) -> Dict[str, float]:
    return {
        "grid_cap_kw": float(data.grid_cap.max()),
        "station_cap_total_kw": float(data.station_cap.sum()),
        "sum_ev_pmax_kw": float(data.p_max.sum()),
        "required_kwh": float(data.R.sum()),
        "grid_energy_kwh": float(data.grid_cap.sum() * data.dt),
    }