"""
Electricity price profiles pi_t [Rs/kWh].

    tou        time-of-use tariff from common/constants.TOU_TARIFF (default)
    flat       constant price = time-average of the TOU tariff
    two_level  peak / off-peak
    real_time  smooth daily curve (peak ~19:00) + random noise (seeded)

Usage
    prices = get_prices("real_time", data.T, data.dt, data.start_hour, seed=1)
    data2  = with_prices(data, prices)
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from common.constants import DT_HOURS, START_HOUR, T_STEPS
from common.utils import ProblemData, clock_hours, tou_prices

PRICE_PROFILES = ("tou", "flat", "two_level", "real_time")


def flat_prices(T: int = T_STEPS, dt: float = DT_HOURS, start_hour: float = START_HOUR,
                price: float | None = None) -> np.ndarray:
    if price is None:
        price = float(np.mean(tou_prices(int(round(24 / dt)), dt, 0.0)))
    return np.full(T, float(price))


def two_level_prices(T: int = T_STEPS, dt: float = DT_HOURS, start_hour: float = START_HOUR,
                     peak_hours: tuple = (17.0, 22.0), high: float = 9.5,
                     low: float = 5.0) -> np.ndarray:
    h = clock_hours(T, dt, start_hour)
    return np.where((h >= peak_hours[0]) & (h < peak_hours[1]), high, low).astype(float)


def real_time_prices(T: int = T_STEPS, dt: float = DT_HOURS, start_hour: float = START_HOUR,
                     seed: int = 0, base: float = 6.5, amplitude: float = 2.5,
                     noise: float = 0.5, peak_hour: float = 19.0) -> np.ndarray:
    h = clock_hours(T, dt, start_hour)
    rng = np.random.default_rng(seed)
    p = base + amplitude * np.cos(2 * np.pi * (h - peak_hour) / 24.0) + rng.normal(0, noise, T)
    return np.clip(p, 1.0, None)


def get_prices(profile: str = "tou", T: int = T_STEPS, dt: float = DT_HOURS,
               start_hour: float = START_HOUR, seed: int = 0, **kwargs) -> np.ndarray:
    if profile == "tou":
        return tou_prices(T, dt, start_hour, **kwargs)
    if profile == "flat":
        return flat_prices(T, dt, start_hour, **kwargs)
    if profile == "two_level":
        return two_level_prices(T, dt, start_hour, **kwargs)
    if profile == "real_time":
        return real_time_prices(T, dt, start_hour, seed=seed, **kwargs)
    raise ValueError(f"unknown price profile '{profile}', choose from {PRICE_PROFILES}")


def with_prices(data: ProblemData, prices: np.ndarray, keep_C_ref: bool = True) -> ProblemData:
    """Copy of ``data`` with new prices. C_ref is kept by default so J_cost
    stays comparable across price profiles."""
    prices = np.asarray(prices, dtype=float)
    if prices.size != data.T:
        raise ValueError("prices must have length T")
    return replace(data, prices=prices, C_ref=data.C_ref if keep_C_ref else None)