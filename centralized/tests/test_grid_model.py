import numpy as np
import pytest

from centralized.grid_model import station_loads, utilization, with_capacity
from centralized.prices import PRICE_PROFILES, get_prices, with_prices
from centralized.uncoordinated import asap_schedule, fcfs_schedule
from ev.fleet import scenario_instance


@pytest.fixture(scope="module")
def data():
    return scenario_instance("morning_peak", 30, 3, seed=0)[0]


def test_with_capacity_scales_and_keeps_refs(data):
    d2 = with_capacity(data, grid_factor=0.5, station_factor=2.0)
    assert np.allclose(d2.grid_cap, 0.5 * data.grid_cap)
    assert np.allclose(d2.station_cap, 2.0 * data.station_cap)
    assert d2.P_ref == data.P_ref and d2.C_ref == data.C_ref
    assert np.array_equal(d2.R, data.R)


def test_station_loads_sum_to_total(data):
    P = asap_schedule(data)
    assert np.allclose(station_loads(data, P).sum(axis=0), P.sum(axis=0))


def test_fcfs_respects_capacity_asap_does_not(data):
    u_fcfs = utilization(data, fcfs_schedule(data))
    u_asap = utilization(data, asap_schedule(data))
    assert u_fcfs["grid_util_max"] <= 1 + 1e-9 and u_fcfs["station_util_max"] <= 1 + 1e-9
    assert u_asap["grid_util_max"] > 1 or u_asap["station_util_max"] > 1


@pytest.mark.parametrize("profile", PRICE_PROFILES)
def test_price_profiles(data, profile):
    p = get_prices(profile, data.T, data.dt, data.start_hour, seed=0)
    assert p.shape == (data.T,) and np.all(p > 0)
    d2 = with_prices(data, p)
    assert np.allclose(d2.prices, p) and d2.C_ref == data.C_ref