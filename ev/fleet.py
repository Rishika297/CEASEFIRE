"""
Generate EV fleets and turn them into a ProblemData instance.

Quick start (from the project root):
    python -m ev.fleet

    from ev.fleet import make_instance
    data, evs = make_instance(n_evs=50, n_stations=3, scenario="morning_peak", seed=0)
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.constants import (BATTERY_CHOICES_KWH, BATTERY_PROBS, DEFAULT_ETA,
                              DEMAND_PROFILES, DT_HOURS, EXP5_SCENARIOS,
                              GRID_CAP_FACTOR, HORIZON_HOURS, P_MAX_CHOICES_KW,
                              P_MAX_PROBS, SOC_MAX, SOC_MIN, START_HOUR,
                              STATION_KW_PER_EV)
from common.utils import ProblemData, load_yaml, tou_prices
from ev.ev_model import EV
from ev.schedule import sample_arrivals_departures


@dataclass
class FleetConfig:
    n_evs: int = 50
    n_stations: int = 3
    scenario: str = "random"            # arrival pattern: morning_peak | evening_peak | random
    demand: str = "normal"              # low | normal | high
    seed: int = 0

    dt: float = DT_HOURS
    horizon_h: float = HORIZON_HOURS
    start_hour: float = START_HOUR

    eta: float = DEFAULT_ETA
    soc_min: float = SOC_MIN
    soc_max: float = SOC_MAX
    battery_choices: Sequence[float] = BATTERY_CHOICES_KWH
    battery_probs: Sequence[float] = BATTERY_PROBS
    p_max_choices: Sequence[float] = P_MAX_CHOICES_KW
    p_max_probs: Sequence[float] = P_MAX_PROBS
    station_assignment: str = "balanced"    # balanced | random

    # station / grid sizing
    station_kw_per_ev: float = STATION_KW_PER_EV
    grid_cap_factor: float = GRID_CAP_FACTOR

    @property
    def T(self) -> int:
        return int(round(self.horizon_h / self.dt))

    @classmethod
    def from_dict(cls, d: Dict) -> "FleetConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown fleet config keys: {sorted(unknown)}")
        return cls(**d)


def load_fleet_config(path: str = "configs/fleet_config.yaml", **overrides) -> FleetConfig:
    d = load_yaml(path).get("fleet", {})
    d.update(overrides)
    return FleetConfig.from_dict(d)


# ----------------------------------------------------------------------
def generate_fleet(cfg: FleetConfig) -> List[EV]:
    if cfg.demand not in DEMAND_PROFILES:
        raise ValueError(f"unknown demand level '{cfg.demand}', choose from {list(DEMAND_PROFILES)}")
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_evs

    arrival, departure = sample_arrivals_departures(
        n, cfg.scenario, rng, dt=cfg.dt, horizon_h=cfg.horizon_h, start_hour=cfg.start_hour)

    battery = rng.choice(cfg.battery_choices, n, p=_norm(cfg.battery_probs))
    p_max = rng.choice(cfg.p_max_choices, n, p=_norm(cfg.p_max_probs))

    dem = DEMAND_PROFILES[cfg.demand]
    soc_init = rng.uniform(*dem["soc_init"], n)
    soc_target = rng.uniform(*dem["soc_target"], n)
    soc_init = np.clip(soc_init, cfg.soc_min, cfg.soc_max)
    soc_target = np.clip(np.maximum(soc_target, soc_init), cfg.soc_min, cfg.soc_max)

    if cfg.station_assignment == "balanced":
        station = rng.permutation(np.arange(n) % cfg.n_stations)
    elif cfg.station_assignment == "random":
        station = rng.integers(0, cfg.n_stations, n)
    else:
        raise ValueError("station_assignment must be 'balanced' or 'random'")

    return [
        EV(ev_id=i, battery_kwh=float(battery[i]), soc_init=float(soc_init[i]),
           soc_target=float(soc_target[i]), arrival=int(arrival[i]),
           departure=int(departure[i]), p_max_kw=float(p_max[i]), eta=cfg.eta,
           station=int(station[i]), soc_min=cfg.soc_min, soc_max=cfg.soc_max)
        for i in range(n)
    ]


def _norm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p / p.sum()


# ----------------------------------------------------------------------
def station_capacities(evs: List[EV], n_stations: int,
                       kw_per_ev: float = STATION_KW_PER_EV) -> np.ndarray:
    """P_m^max = kw_per_ev * (#EVs assigned to station m), at least kw_per_ev."""
    counts = np.bincount([ev.station for ev in evs], minlength=n_stations)
    return kw_per_ev * np.maximum(counts, 1).astype(float)


def build_problem_data(evs: List[EV], cfg: FleetConfig,
                       prices: Optional[np.ndarray] = None,
                       station_cap: Optional[np.ndarray] = None,
                       grid_cap: Optional[float | np.ndarray] = None,
                       **refs) -> ProblemData:
    """Convert a list of EVs into the array-based ProblemData.
    ``refs`` may contain C_ref, P_ref, H_ref."""
    T = cfg.T
    if prices is None:
        prices = tou_prices(T, cfg.dt, cfg.start_hour)
    if station_cap is None:
        station_cap = station_capacities(evs, cfg.n_stations, cfg.station_kw_per_ev)
    if grid_cap is None:
        grid_cap = cfg.grid_cap_factor * float(np.sum(station_cap))
    a = lambda attr: np.array([getattr(ev, attr) for ev in evs])
    return ProblemData(
        T=T, dt=cfg.dt, prices=prices, grid_cap=grid_cap, station_cap=station_cap,
        station=a("station"), battery=a("battery_kwh"), soc_init=a("soc_init"),
        soc_target=a("soc_target"), eta=a("eta"), p_max=a("p_max_kw"),
        arrival=a("arrival"), departure=a("departure"),
        soc_min=a("soc_min"), soc_max=a("soc_max"), ev_ids=a("ev_id"),
        start_hour=cfg.start_hour, **refs,
    )


def make_instance(n_evs: int = 50, n_stations: int = 3, scenario: str = "random",
                  demand: str = "normal", seed: int = 0,
                  **overrides) -> Tuple[ProblemData, List[EV]]:
    """One-liner used by every experiment."""
    cfg = FleetConfig(n_evs=n_evs, n_stations=n_stations, scenario=scenario,
                      demand=demand, seed=seed, **overrides)
    evs = generate_fleet(cfg)
    return build_problem_data(evs, cfg), evs


def scenario_instance(name: str, n_evs: int = 50, n_stations: int = 3, seed: int = 0,
                      **overrides) -> Tuple[ProblemData, List[EV]]:
    """Experiment 5 scenarios: morning_peak, evening_peak, random, high_demand, low_demand."""
    if name not in EXP5_SCENARIOS:
        raise ValueError(f"unknown scenario '{name}', choose from {list(EXP5_SCENARIOS)}")
    arrival, demand = EXP5_SCENARIOS[name]
    return make_instance(n_evs, n_stations, arrival, demand, seed, **overrides)


def fleet_summary(evs: List[EV], dt: float) -> Dict[str, float]:
    R = np.array([ev.required_energy_kwh for ev in evs])
    stay = np.array([ev.stay_hours(dt) for ev in evs])
    feas = np.array([ev.is_individually_feasible(dt) for ev in evs])
    return {
        "n_evs": len(evs),
        "total_required_kwh": float(R.sum()),
        "mean_required_kwh": float(R.mean()),
        "mean_stay_h": float(stay.mean()),
        "pct_individually_feasible": float(100 * feas.mean()),
    }


if __name__ == "__main__":
    for name in EXP5_SCENARIOS:
        data, evs = scenario_instance(name, n_evs=50, n_stations=3, seed=0)
        print(f"{name:13s}", fleet_summary(evs, data.dt),
              {k: round(v, 2) for k, v in data.summary().items() if k in ("grid_cap_kw", "station_cap_kw")})
