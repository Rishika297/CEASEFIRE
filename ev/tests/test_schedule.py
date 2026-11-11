import numpy as np
import pytest

from ev.schedule import (active_at, availability_matrix, events, occupancy,
                         sample_arrivals_departures)

T, DT = 48, 0.5


@pytest.mark.parametrize("scenario", ["morning_peak", "evening_peak", "random"])
def test_arrivals_departures_valid(scenario):
    a, d = sample_arrivals_departures(500, scenario, np.random.default_rng(0), dt=DT)
    assert np.all(a >= 0) and np.all(a < T)
    assert np.all(d > a) and np.all(d <= T)


def test_morning_before_evening():
    rng = np.random.default_rng(1)
    am, _ = sample_arrivals_departures(500, "morning_peak", rng, dt=DT)
    pm, _ = sample_arrivals_departures(500, "evening_peak", rng, dt=DT)
    assert am.mean() < pm.mean()


def test_unknown_scenario():
    with pytest.raises(ValueError):
        sample_arrivals_departures(5, "lunch", np.random.default_rng(0))


def test_occupancy_matches_stays():
    a, d = sample_arrivals_departures(100, "random", np.random.default_rng(2), dt=DT)
    assert occupancy(a, d, T).sum() == np.sum(d - a)
    assert availability_matrix(a, d, T).shape == (100, T)
    t = int(a[0])
    assert 0 in active_at(t, a, d)


def test_events_sorted():
    ev = events(np.array([0, 2]), np.array([2, 5]))
    assert [e[0] for e in ev] == sorted(e[0] for e in ev)
    assert ev[1][1] == "depart" and ev[2][1] == "arrive"      # depart before arrive at t=2
