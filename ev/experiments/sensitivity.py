"""
Battery, power and SOC sensitivity (EV level, no coordination).

Sweeps one parameter at a time and reports
  * % EVs individually feasible
  * mean minimum charging time [h]
  * total required energy [kWh]

Run:  python -m ev.experiments.sensitivity
Out:  results/scenarios/ev_sensitivity.csv, plots/scenarios/ev_sensitivity.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.utils import plots_dir, results_dir
from ev.fleet import FleetConfig, fleet_summary, generate_fleet

SWEEPS = {
    "battery_kwh": [30, 40, 60, 75, 100],
    "p_max_kw":    [3.3, 7.4, 11.0, 22.0],
    "soc_init":    [0.05, 0.2, 0.35, 0.5, 0.65],
}


def evaluate(cfg: FleetConfig, param: str, value: float) -> dict:
    evs = generate_fleet(cfg)
    for ev in evs:                                   # override the swept parameter
        if param == "battery_kwh":
            ev.battery_kwh = float(value)
        elif param == "p_max_kw":
            ev.p_max_kw = float(value)
        elif param == "soc_init":
            ev.soc_init = float(value)
            ev.soc_target = max(ev.soc_target, ev.soc_init)
    summ = fleet_summary(evs, cfg.dt)
    summ["mean_min_charge_h"] = float(np.mean([ev.min_charging_steps(cfg.dt) * cfg.dt for ev in evs]))
    summ.update({"param": param, "value": value})
    return summ


def run(n_evs: int = 100, scenario: str = "random", seeds=(0, 1, 2)) -> pd.DataFrame:
    rows = [evaluate(FleetConfig(n_evs=n_evs, scenario=scenario, seed=s), p, v)
            for p, vals in SWEEPS.items() for v in vals for s in seeds]
    df = pd.DataFrame(rows).groupby(["param", "value"]).mean(numeric_only=True).reset_index()
    df.to_csv(results_dir("scenarios") / "ev_sensitivity.csv", index=False)

    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for j, p in enumerate(SWEEPS):
        d = df[df.param == p]
        ax[0, j].plot(d.value, d.pct_individually_feasible, "o-")
        ax[0, j].set_title(p); ax[0, j].set_ylabel("% individually feasible")
        ax[1, j].plot(d.value, d.mean_min_charge_h, "s-", color="C1")
        ax[1, j].set_ylabel("mean min. charging time [h]"); ax[1, j].set_xlabel(p)
    fig.tight_layout()
    out = plots_dir("scenarios") / "ev_sensitivity.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(df.round(2).to_string(index=False))
    print(f"saved -> {out}")
    return df


if __name__ == "__main__":
    run()
