"""Config validation + tiny smoke runs of every experiment runner
(outputs are written with the suffix _smoke so real results are not overwritten)."""
import pytest

cp = pytest.importorskip("cvxpy")

from algorithms.base import ADMMSettings
from common.objective import WEIGHT_CASES
from evaluation.objective_evaluation import DEFAULT_RUNS
from ev.fleet import load_fleet_config
from network.integration import load_network_config


def test_configs_load():
    st = ADMMSettings.load()
    assert st.max_iter > 0 and st.mu > 0 and st.mu_consensus > 0
    assert load_fleet_config().n_evs > 0
    assert "averaging" in load_network_config()


def test_exp2_smoke():
    from experiments.objective_weight_sensitivity import EXP2_RUNS, run
    df = run(n_evs=8, n_stations=2, seeds=(0,), max_iter=30, tag="_smoke")
    assert len(df) == len(WEIGHT_CASES) * len(EXP2_RUNS)
    cen = df[df.algorithm == "centralized"]
    assert (cen["gap"] == 0).all()


def test_exp3_smoke():
    from experiments.scalability import PAIRS, run
    df = run(cases=((8, 1), (12, 2)), seeds=(0,), max_iter=30, tag="_smoke")
    assert len(df) == 2 * (1 + len(PAIRS))
    assert set(df["n_evs"]) == {8, 12}


def test_exp5_smoke():
    from common.constants import EXP5_SCENARIOS
    from experiments.charging_scenarios import run
    df = run(n_evs=8, n_stations=2, seeds=(0,), max_iter=30, tag="_smoke")
    assert len(df) == len(EXP5_SCENARIOS) * len(DEFAULT_RUNS)