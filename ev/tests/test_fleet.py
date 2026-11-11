import numpy as np
import pytest

from common.constants import EXP5_SCENARIOS, SCALABILITY_CASES
from ev.fleet import FleetConfig, generate_fleet, make_instance, scenario_instance


def test_reproducible():
    a = generate_fleet(FleetConfig(n_evs=20, seed=3))
    b = generate_fleet(FleetConfig(n_evs=20, seed=3))
    assert [e.to_dict() for e in a] == [e.to_dict() for e in b]


@pytest.mark.parametrize("n_evs,n_st", SCALABILITY_CASES)
def test_instance_shapes(n_evs, n_st):
    data, evs = make_instance(n_evs, n_st, seed=0)
    assert data.N == n_evs and data.M == n_st and len(evs) == n_evs
    assert data.P_ref > 0 and data.C_ref > 0
    assert np.all(data.station_cap > 0)
    assert np.all(np.bincount(data.station, minlength=n_st) > 0)   # balanced assignment


@pytest.mark.parametrize("name", list(EXP5_SCENARIOS))
def test_scenarios(name):
    data, _ = scenario_instance(name, 30, 3, seed=0)
    assert np.all(data.R >= 0) and np.all(data.soc_target <= data.soc_max)


def test_demand_levels_ordered():
    lo, _ = scenario_instance("low_demand", 200, 5, seed=0)
    hi, _ = scenario_instance("high_demand", 200, 5, seed=0)
    assert lo.R.sum() < hi.R.sum()
