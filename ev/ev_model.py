"""
EV model and battery specification.

EV        : static description of one charging session (what the agent knows).
EVState   : mutable state used when stepping a simulation forward in time
            (dynamic arrivals / rolling horizon).

Time is in integer steps of length dt [h]; the EV is plugged in for
t in [arrival, departure)  (departure exclusive).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from common.constants import DEFAULT_ETA, NUM_TOL, SOC_MAX, SOC_MIN


@dataclass
class EV:
    ev_id: int
    battery_kwh: float          # B_i
    soc_init: float             # SOC_i^initial
    soc_target: float           # SOC_i^target
    arrival: int                # a_i (step index)
    departure: int              # d_i (step index, exclusive)
    p_max_kw: float             # P_i^max (grid side)
    eta: float = DEFAULT_ETA    # eta_i
    station: int = 0            # m(i)
    soc_min: float = SOC_MIN
    soc_max: float = SOC_MAX

    def __post_init__(self) -> None:
        if self.battery_kwh <= 0:
            raise ValueError(f"EV {self.ev_id}: battery_kwh must be > 0")
        if self.p_max_kw <= 0:
            raise ValueError(f"EV {self.ev_id}: p_max_kw must be > 0")
        if not 0 < self.eta <= 1:
            raise ValueError(f"EV {self.ev_id}: eta must be in (0, 1]")
        if not 0 <= self.soc_min <= self.soc_max <= 1:
            raise ValueError(f"EV {self.ev_id}: need 0 <= soc_min <= soc_max <= 1")
        if not self.soc_min <= self.soc_init <= self.soc_max:
            raise ValueError(f"EV {self.ev_id}: soc_init outside [soc_min, soc_max]")
        if not 0 <= self.soc_target <= self.soc_max:
            raise ValueError(f"EV {self.ev_id}: soc_target must be in [0, soc_max]")
        if self.arrival < 0 or self.departure <= self.arrival:
            raise ValueError(f"EV {self.ev_id}: need 0 <= arrival < departure")
        self.arrival, self.departure = int(self.arrival), int(self.departure)

    # ------------------------------------------------------------------
    # Energy quantities
    # ------------------------------------------------------------------
    @property
    def required_energy_kwh(self) -> float:
        """R_i = B_i (SOC_target - SOC_init), battery side, >= 0."""
        return self.battery_kwh * max(self.soc_target - self.soc_init, 0.0)

    @property
    def required_grid_energy_kwh(self) -> float:
        """Energy drawn from the grid to deliver R_i."""
        return self.required_energy_kwh / self.eta

    @property
    def headroom_kwh(self) -> float:
        """Max battery-side energy before hitting SOC_max."""
        return self.battery_kwh * (self.soc_max - self.soc_init)

    @property
    def stay_steps(self) -> int:
        return self.departure - self.arrival

    def stay_hours(self, dt: float) -> float:
        return self.stay_steps * dt

    def max_deliverable_kwh(self, dt: float) -> float:
        """Battery-side energy if charged at P_max for the whole stay (no coupling)."""
        return min(self.eta * self.p_max_kw * self.stay_steps * dt, self.headroom_kwh)

    def min_charging_steps(self, dt: float) -> int:
        """Steps needed at full power to deliver R_i."""
        R = self.required_energy_kwh
        return 0 if R <= NUM_TOL else math.ceil(R / (self.eta * self.p_max_kw * dt) - 1e-12)

    def is_individually_feasible(self, dt: float) -> bool:
        """Can the EV reach its target if it had the charger to itself?"""
        return self.max_deliverable_kwh(dt) >= self.required_energy_kwh - 1e-9

    def laxity_steps(self, dt: float) -> int:
        """Slack in the schedule: stay - minimum charging time (negative = infeasible)."""
        return self.stay_steps - self.min_charging_steps(dt)

    # ------------------------------------------------------------------
    # Time-indexed quantities
    # ------------------------------------------------------------------
    def availability(self, T: int) -> np.ndarray:
        """A_{i,t} for t = 0..T-1 (bool)."""
        t = np.arange(T)
        return (t >= self.arrival) & (t < min(self.departure, T))

    def power_upper(self, T: int) -> np.ndarray:
        """P_i^max * A_{i,t}."""
        return self.p_max_kw * self.availability(T)

    def soc_trajectory(self, p: np.ndarray, dt: float) -> np.ndarray:
        """SOC at the start of every interval plus final SOC -> length T+1."""
        inc = self.eta * np.asarray(p, dtype=float) * dt / self.battery_kwh
        return self.soc_init + np.concatenate([[0.0], np.cumsum(inc)])

    def delivered_energy_kwh(self, p: np.ndarray, dt: float) -> float:
        return float(self.eta * dt * np.sum(p))

    def asap_schedule(self, T: int, dt: float, p_limit: Optional[float] = None) -> np.ndarray:
        """Uncoordinated behaviour: charge at full power from arrival until the
        target (or SOC_max) is reached; last interval is partial."""
        p_cap = self.p_max_kw if p_limit is None else min(self.p_max_kw, p_limit)
        need = min(self.required_energy_kwh, self.headroom_kwh)
        p = np.zeros(T)
        for t in range(self.arrival, min(self.departure, T)):
            if need <= NUM_TOL:
                break
            pt = min(p_cap, need / (self.eta * dt))
            p[t] = pt
            need -= self.eta * pt * dt
        return p

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EV":
        return cls(**d)


# ----------------------------------------------------------------------
class EVState:
    """Mutable battery state for step-by-step simulation."""

    def __init__(self, ev: EV):
        self.ev = ev
        self.soc = ev.soc_init
        self.delivered_kwh = 0.0
        self.history: list[float] = []

    def is_plugged(self, t: int) -> bool:
        return self.ev.arrival <= t < self.ev.departure

    @property
    def remaining_kwh(self) -> float:
        return max(self.ev.battery_kwh * (self.ev.soc_target - self.soc), 0.0)

    def step(self, p_kw: float, t: int, dt: float) -> float:
        """Apply power p for one interval; clips to the physical limits and
        returns the power actually applied."""
        ev = self.ev
        if not self.is_plugged(t):
            p = 0.0
        else:
            room = ev.battery_kwh * (ev.soc_max - self.soc)            # kWh
            p = float(np.clip(p_kw, 0.0, ev.p_max_kw))
            p = min(p, max(room, 0.0) / (ev.eta * dt))
        e = ev.eta * p * dt
        self.soc += e / ev.battery_kwh
        self.delivered_kwh += e
        self.history.append(p)
        return p

    def reached_target(self, tol: float = 1e-3) -> bool:
        return self.soc >= self.ev.soc_target - tol
