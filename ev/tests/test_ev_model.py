import numpy as np
import pytest

from ev.ev_model import EV, EVState

DT, T = 0.5, 48


def make_ev(**kw):
    d = dict(ev_id=0, battery_kwh=40.0, soc_init=0.2, soc_target=0.8,
             arrival=4, departure=20, p_max_kw=7.4, eta=0.9)
    d.update(kw)
    return EV(**d)


def test_required_energy():
    ev = make_ev()
    assert ev.required_energy_kwh == pytest.approx(40 * 0.6)
    assert make_ev(soc_init=0.9, soc_target=0.8).required_energy_kwh == 0.0


@pytest.mark.parametrize("bad", [dict(battery_kwh=0), dict(eta=1.5), dict(departure=4),
                                 dict(soc_init=1.2), dict(p_max_kw=-1)])
def test_validation(bad):
    with pytest.raises(ValueError):
        make_ev(**bad)


def test_asap_reaches_target_and_respects_window():
    ev = make_ev()
    p = ev.asap_schedule(T, DT)
    assert ev.delivered_energy_kwh(p, DT) == pytest.approx(ev.required_energy_kwh)
    assert np.all(p[: ev.arrival] == 0) and np.all(p[ev.departure:] == 0)
    assert p.max() <= ev.p_max_kw + 1e-12


def test_soc_trajectory_end_value():
    ev = make_ev()
    p = ev.asap_schedule(T, DT)
    soc = ev.soc_trajectory(p, DT)
    assert soc.shape == (T + 1,)
    assert soc[0] == pytest.approx(ev.soc_init)
    assert soc[-1] == pytest.approx(ev.soc_target)


def test_individual_feasibility():
    assert make_ev().is_individually_feasible(DT)
    assert not make_ev(departure=6, p_max_kw=3.3).is_individually_feasible(DT)


def test_ev_state_clips_power():
    ev = make_ev(soc_init=0.95, soc_target=0.95, soc_max=1.0)
    st = EVState(ev)
    applied = st.step(100.0, t=ev.arrival, dt=DT)
    assert applied <= ev.p_max_kw
    assert st.soc <= 1.0 + 1e-12
    assert st.step(5.0, t=0, dt=DT) == 0.0          # not plugged in
