"""
Shared helpers and the ProblemData container used by every module.

ProblemData
-----------
All EVs are stored in flat arrays of length N (index i).  The station an EV is
plugged into is ``station[i]`` (index m), so the set E_m of the write-up is
``np.where(station == m)``.  Charging powers are an (N, T) matrix P with
P[i, t] = p_{m(i), i, t}.

Time convention
* t = 0, ..., T-1 are intervals of length ``dt`` hours.
* EV i is plugged in for t in [arrival[i], departure[i])  (departure exclusive).
* SOC_{i,t} is the state of charge at the *start* of interval t, so
  SOC_{i,0} = soc_init[i] and SOC_{i,T} is the final SOC.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from common.constants import DT_HOURS, H_REF_HOURS, START_HOUR, TOU_TARIFF

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ======================================================================
# Generic helpers
# ======================================================================
def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    import yaml  # pyyaml
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(sub: str = "") -> Path:
    return ensure_dir(PROJECT_ROOT / "results" / sub)


def plots_dir(sub: str = "") -> Path:
    return ensure_dir(PROJECT_ROOT / "plots" / sub)


def save_json(obj: Any, path: str | Path) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError(type(o))
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)


def hours_to_steps(hours: float, dt: float = DT_HOURS) -> int:
    return int(round(hours / dt))


def clock_hours(T: int, dt: float = DT_HOURS, start_hour: float = START_HOUR) -> np.ndarray:
    """Clock hour (0-24) at the start of each interval."""
    return (start_hour + np.arange(T) * dt) % 24.0


def tou_prices(T: int, dt: float = DT_HOURS, start_hour: float = START_HOUR,
               tariff: Sequence = TOU_TARIFF) -> np.ndarray:
    """Time-of-use price pi_t [Rs/kWh] for each interval (Himani's
    centralized/prices.py can replace this with other profiles)."""
    h = clock_hours(T, dt, start_hour)
    prices = np.empty(T)
    for k, hk in enumerate(h):
        for lo, hi, price in tariff:
            if lo <= hk < hi:
                prices[k] = price
                break
        else:
            raise ValueError(f"tariff does not cover hour {hk}")
    return prices


class Timer:
    """with Timer() as tm: ...;  tm.elapsed -> seconds"""
    def __enter__(self):
        self._t0 = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
        return False


# ======================================================================
# Problem container
# ======================================================================
@dataclass
class ProblemData:
    # ---------------- horizon / grid ----------------
    T: int                              # number of time intervals
    dt: float                           # interval length [h]
    prices: np.ndarray                  # (T,) electricity price pi_t [Rs/kWh]
    grid_cap: np.ndarray                # (T,) or scalar, P_grid^max [kW]
    station_cap: np.ndarray             # (M,) P_m^max [kW]

    # ---------------- EVs (length N) ----------------
    station: np.ndarray                 # station index m(i) in {0..M-1}
    battery: np.ndarray                 # B_i [kWh]
    soc_init: np.ndarray                # SOC_i^initial in [0, 1]
    soc_target: np.ndarray              # SOC_i^target in [0, 1]
    eta: np.ndarray                     # charging efficiency eta_i in (0, 1]
    p_max: np.ndarray                   # P_i^max, grid-side [kW]
    arrival: np.ndarray                 # a_i (int)
    departure: np.ndarray               # d_i (int, exclusive)
    soc_min: Optional[np.ndarray] = None    # default 0
    soc_max: Optional[np.ndarray] = None    # default 1

    # ---------------- normalisation references ----------------
    C_ref: Optional[float] = None       # [Rs]  default: cost of all required energy at mean price
    P_ref: Optional[float] = None       # [kW]  default: max grid capacity
    H_ref: float = H_REF_HOURS          # [h]

    ev_ids: Optional[np.ndarray] = None
    start_hour: float = START_HOUR      # only used for plotting clock time

    # ---------------- derived (filled in __post_init__) ----------------
    N: int = field(init=False)
    M: int = field(init=False)
    R: np.ndarray = field(init=False)                # required battery-side energy [kWh]
    inv_R: np.ndarray = field(init=False)            # 1/R_i, 0 when R_i == 0
    avail: np.ndarray = field(init=False)            # (N, T) bool, A_{i,t}
    p_upper: np.ndarray = field(init=False)          # (N, T) P_i^max * A_{i,t}
    energy_headroom: np.ndarray = field(init=False)  # B_i (SOC_max - SOC_init) [kWh]
    delay_weight: np.ndarray = field(init=False)     # (N, T) coefficient of p in J_delay
    station_matrix: np.ndarray = field(init=False)   # (M, N) 0/1 incidence
    delay_const: float = field(init=False)
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        f = lambda x: np.asarray(x, dtype=float).ravel()
        self.prices = f(self.prices)
        self.station_cap = f(self.station_cap)
        self.battery = f(self.battery)
        self.soc_init = f(self.soc_init)
        self.soc_target = f(self.soc_target)
        self.eta = f(self.eta)
        self.p_max = f(self.p_max)
        self.station = np.asarray(self.station, dtype=int).ravel()
        self.arrival = np.asarray(self.arrival, dtype=int).ravel()
        self.departure = np.asarray(self.departure, dtype=int).ravel()

        self.N = self.battery.size
        self.M = self.station_cap.size
        self.soc_min = np.zeros(self.N) if self.soc_min is None else f(self.soc_min)
        self.soc_max = np.ones(self.N) if self.soc_max is None else f(self.soc_max)
        gc = np.asarray(self.grid_cap, dtype=float)
        self.grid_cap = np.full(self.T, float(gc)) if gc.ndim == 0 else gc.ravel()
        if self.ev_ids is None:
            self.ev_ids = np.arange(self.N)

        self._validate()
        self._derive()

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        N, T = self.N, self.T
        if T <= 0 or self.dt <= 0:
            raise ValueError("T and dt must be positive")
        for name in ("soc_init", "soc_target", "eta", "p_max", "arrival",
                     "departure", "station", "soc_min", "soc_max"):
            if getattr(self, name).size != N:
                raise ValueError(f"{name} has length {getattr(self, name).size}, expected N={N}")
        if self.prices.size != T or self.grid_cap.size != T:
            raise ValueError("prices and grid_cap must have length T")
        if np.any(self.station < 0) or np.any(self.station >= self.M):
            raise ValueError("station index out of range")
        if np.any(self.battery <= 0) or np.any(self.p_max <= 0):
            raise ValueError("battery capacity and p_max must be positive")
        if np.any(self.eta <= 0) or np.any(self.eta > 1):
            raise ValueError("eta must be in (0, 1]")
        if np.any(self.station_cap < 0) or np.any(self.grid_cap < 0):
            raise ValueError("capacities must be non-negative")
        if np.any(self.soc_min < 0) or np.any(self.soc_max > 1) or np.any(self.soc_min > self.soc_max):
            raise ValueError("need 0 <= soc_min <= soc_max <= 1")
        # ADDED: initial SOC must already satisfy the battery limits
        if np.any(self.soc_init < self.soc_min) or np.any(self.soc_init > self.soc_max):
            raise ValueError("soc_init must lie in [soc_min, soc_max]")
        # ADDED: target must be reachable without breaking SOC_max
        if np.any(self.soc_target > self.soc_max):
            raise ValueError("soc_target must be <= soc_max")
        if np.any(self.arrival < 0) or np.any(self.arrival >= T):
            raise ValueError("arrival must be in [0, T-1]")
        if np.any(self.departure <= self.arrival):
            raise ValueError("departure must be > arrival")
        # ADDED: departures beyond the horizon are truncated to T
        if np.any(self.departure > T):
            warnings.warn("some departures exceed the horizon; truncated to T")
            self.departure = np.minimum(self.departure, T)

    # ------------------------------------------------------------------
    def _derive(self) -> None:
        N, T, dt = self.N, self.T, self.dt
        t = np.arange(T)

        # R_i = B_i (SOC_target - SOC_init); EVs already above target need 0.
        self.R = self.battery * np.maximum(self.soc_target - self.soc_init, 0.0)
        self.inv_R = np.divide(1.0, self.R, out=np.zeros(N), where=self.R > 1e-9)

        # A_{i,t} = 1 while the EV is plugged in
        self.avail = (t[None, :] >= self.arrival[:, None]) & (t[None, :] < self.departure[:, None])
        self.p_upper = self.p_max[:, None] * self.avail

        # C3 + C9: cannot exceed SOC_max, and never charge beyond the target R_i
        self.energy_headroom = np.minimum(self.battery * (self.soc_max - self.soc_init), self.R)

        # J_delay = sum_i sum_t W[i,t] p[i,t],
        # W[i,t] = (t - a_i) dt / H_ref * eta_i dt / R_i  on the availability window
        wait = (t[None, :] - self.arrival[:, None]) * dt / self.H_ref
        # J_delay = delay_const + sum_{i,t} W[i,t] p[i,t]   (undelivered-energy form)
        #   W[i,t]      = -(d_i - t) dt / H_ref * eta_i dt / R_i   on the availability window
        #   delay_const = sum_i (d_i - a_i) dt / H_ref            (EVs with R_i > 0)
        remaining = (self.departure[:, None] - t[None, :]) * dt / self.H_ref
        self.delay_weight = -np.where(self.avail, remaining, 0.0) * (self.eta * dt * self.inv_R)[:, None]
        self.delay_const = float(np.sum(((self.departure - self.arrival) * dt / self.H_ref)[self.R > 1e-9]))

        self.station_matrix = np.zeros((self.M, N))
        self.station_matrix[self.station, np.arange(N)] = 1.0

        if self.C_ref is None:
            c = float(np.mean(self.prices) * np.sum(self.R / self.eta))
            self.C_ref = c if c > 0 else 1.0
        if self.P_ref is None:
            p = float(np.max(self.grid_cap))
            if not np.isfinite(p) or p <= 0:
                p = float(np.sum(self.station_cap)) or 1.0
            self.P_ref = p
        n_req = max(int(np.sum(self.R > 1e-9)), 1)
       
        self.delay_weight /= n_req
        self.delay_const /= n_req

    # ------------------------------------------------------------------
    def evs_at(self, m: int) -> np.ndarray:
        """Indices of EVs in E_m."""
        return np.flatnonzero(self.station == m)

    def subset(self, idx) -> "ProblemData":
        """New instance with only the EVs in ``idx`` (dynamic arrivals / tests).
        Keeps the same C_ref/P_ref/H_ref so objectives stay comparable."""
        idx = np.asarray(idx)
        return ProblemData(
            T=self.T, dt=self.dt, prices=self.prices, grid_cap=self.grid_cap,
            station_cap=self.station_cap, station=self.station[idx],
            battery=self.battery[idx], soc_init=self.soc_init[idx],
            soc_target=self.soc_target[idx], eta=self.eta[idx], p_max=self.p_max[idx],
            arrival=self.arrival[idx], departure=self.departure[idx],
            soc_min=self.soc_min[idx], soc_max=self.soc_max[idx],
            C_ref=self.C_ref, P_ref=self.P_ref, H_ref=self.H_ref,
            ev_ids=self.ev_ids[idx], start_hour=self.start_hour,
        )

    def summary(self) -> Dict[str, float]:
        return {
            "N": self.N, "M": self.M, "T": self.T, "dt": self.dt,
            "total_required_kwh": float(self.R.sum()),
            "grid_cap_kw": float(self.grid_cap.max()),
            "station_cap_kw": float(self.station_cap.sum()),
            "C_ref": float(self.C_ref), "P_ref": float(self.P_ref), "H_ref": float(self.H_ref),
        }


# ----------------------------------------------------------------------
def random_instance(N: int = 20, M: int = 2, T: int = 48, dt: float = 0.5,
                    seed: int = 0, grid_cap: Optional[float] = None) -> ProblemData:
    """Tiny random instance for unit tests. For real scenarios use
    ev.fleet.make_instance()."""
    rng = np.random.default_rng(seed)
    battery = rng.choice([30.0, 40.0, 60.0, 75.0], N)
    soc_init = rng.uniform(0.1, 0.5, N)
    soc_target = np.minimum(soc_init + rng.uniform(0.3, 0.5, N), 0.95)
    arrival = rng.integers(0, T // 2, N)
    departure = np.minimum(arrival + rng.integers(int(3 / dt), int(10 / dt), N), T)
    station = rng.integers(0, M, N)
    p_max = rng.choice([7.4, 11.0], N)
    station_cap = np.array([max(1, np.sum(station == m)) * 7.0 for m in range(M)])
    if grid_cap is None:
        grid_cap = 0.8 * station_cap.sum()
    return ProblemData(T=T, dt=dt, prices=tou_prices(T, dt), grid_cap=grid_cap,
                       station_cap=station_cap, station=station, battery=battery,
                       soc_init=soc_init, soc_target=soc_target,
                       eta=np.full(N, 0.92), p_max=p_max,
                       arrival=arrival, departure=departure)
