import numpy as np

from ev.constraints import (check_ev_schedule, is_ev_feasible, project_box_energy,
                            project_power, unmet_energy)
from ev.ev_model import EV

DT, T = 0.5, 48
EVX = EV(ev_id=0, battery_kwh=40.0, soc_init=0.5, soc_target=0.8,
         arrival=4, departure=20, p_max_kw=7.4, eta=0.9, soc_max=0.9)


def test_asap_is_feasible():
    p = EVX.asap_schedule(T, DT)
    assert is_ev_feasible(EVX, p, DT)
    assert unmet_energy(EVX, p, DT) < 1e-9


def test_detects_power_outside_window():
    p = np.zeros(T); p[0] = 3.0
    assert check_ev_schedule(EVX, p, DT)["power_upper"] > 0


def test_detects_soc_overflow():
    p = EVX.power_upper(T)                     # full power for whole stay -> above soc_max
    assert check_ev_schedule(EVX, p, DT)["soc_max"] > 0


def test_projection_box_energy():
    rng = np.random.default_rng(0)
    v = rng.normal(3, 4, T); ub = np.full(T, 5.0)
    p = project_box_energy(v, ub, e_max=20.0)
    assert np.all(p >= 0) and np.all(p <= ub) and p.sum() <= 20.0 + 1e-8


def test_project_power_gives_feasible_profile():
    p = project_power(EVX, np.full(T, 50.0), T, DT)
    v = check_ev_schedule(EVX, p, DT)
    assert v["power_upper"] < 1e-9 and v["soc_max"] < 1e-8
