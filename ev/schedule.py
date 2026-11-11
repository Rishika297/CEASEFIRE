"""
Arrival, departure and availability.

Arrival patterns (see common/constants.ARRIVAL_PROFILES)
    morning_peak : arrival clock time ~ N(9:00, 1 h),  stay ~ N(8 h, 1.5 h)
    evening_peak : arrival clock time ~ N(18:30, 1.5 h), stay ~ N(10 h, 2 h)
    random       : arrival uniform over the horizon,    stay ~ N(6 h, 2.5 h)

Clock time is converted to a step index relative to the simulation start
(START_HOUR = 06:00 by default).  Departures past the horizon are cut to T.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.constants import (ARRIVAL_PROFILES, DT_HOURS, HORIZON_HOURS,
                              MIN_STAY_HOURS, START_HOUR)


def sample_arrivals_departures(
    n: int,
    scenario: str,
    rng: np.random.Generator,
    dt: float = DT_HOURS,
    horizon_h: float = HORIZON_HOURS,
    start_hour: float = START_HOUR,
    profile: Optional[Dict] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return integer arrays (arrival, departure) of length n."""
    if profile is None:
        if scenario not in ARRIVAL_PROFILES:
            raise ValueError(f"unknown arrival scenario '{scenario}', "
                             f"choose from {list(ARRIVAL_PROFILES)}")
        profile = ARRIVAL_PROFILES[scenario]
    T = int(round(horizon_h / dt))
    min_stay = max(1, math.ceil(MIN_STAY_HOURS / dt))
    last_arrival_h = horizon_h - MIN_STAY_HOURS

    if profile.get("arrival_mean_h") is None:                  # uniform arrivals
        offset_h = rng.uniform(0.0, last_arrival_h, n)
    else:
        clock = rng.normal(profile["arrival_mean_h"], profile["arrival_std_h"], n)
        offset_h = np.clip(clock - start_hour, 0.0, last_arrival_h)

    arrival = np.floor(offset_h / dt).astype(int)
    stay_h = np.maximum(rng.normal(profile["stay_mean_h"], profile["stay_std_h"], n),
                        MIN_STAY_HOURS)
    stay = np.maximum(np.ceil(stay_h / dt).astype(int), min_stay)
    departure = np.minimum(arrival + stay, T)
    return arrival, departure


def availability_matrix(arrival: np.ndarray, departure: np.ndarray, T: int) -> np.ndarray:
    """(N, T) bool matrix A[i, t] = 1 if EV i is plugged in during t."""
    t = np.arange(T)
    return (t[None, :] >= np.asarray(arrival)[:, None]) & (t[None, :] < np.asarray(departure)[:, None])


def occupancy(arrival: np.ndarray, departure: np.ndarray, T: int) -> np.ndarray:
    """Number of EVs plugged in at each interval -> (T,)."""
    return availability_matrix(arrival, departure, T).sum(axis=0)


def active_at(t: int, arrival: np.ndarray, departure: np.ndarray) -> np.ndarray:
    """Indices of EVs plugged in during interval t."""
    return np.flatnonzero((np.asarray(arrival) <= t) & (t < np.asarray(departure)))


def events(arrival: np.ndarray, departure: np.ndarray) -> List[Tuple[int, str, int]]:
    """Time-ordered list of (step, 'arrive' | 'depart', ev_index) for dynamic simulations.
    Departures at the same step are listed before arrivals."""
    ev = [(int(d), "depart", i) for i, d in enumerate(departure)]
    ev += [(int(a), "arrive", i) for i, a in enumerate(arrival)]
    return sorted(ev, key=lambda x: (x[0], 0 if x[1] == "depart" else 1, x[2]))


def step_to_clock(t: int | np.ndarray, dt: float = DT_HOURS, start_hour: float = START_HOUR):
    """Clock hour of step t (e.g. for plot tick labels)."""
    return (start_hour + np.asarray(t) * dt) % 24.0
