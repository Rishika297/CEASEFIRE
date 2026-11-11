"""
EV feasibility experiments.

For every Experiment 3 size and every Experiment 5 scenario reports
  * % EVs that could reach target alone (charger to themselves)
  * demand/supply ratio = total required energy / energy the grid cap can deliver
    while EVs are plugged in (> 1 means some unmet energy is unavoidable)

Run:  python -m ev.experiments.feasibility_check
Out:  results/scenarios/ev_feasibility.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.constants import EXP5_SCENARIOS, SCALABILITY_CASES
from common.utils import results_dir
from ev.fleet import fleet_summary, scenario_instance
from ev.schedule import occupancy


def supply_bound_kwh(data) -> float:
    """Upper bound on deliverable battery-side energy given all capacities."""
    occ = occupancy(data.arrival, data.departure, data.T)
    per_t = np.minimum(data.grid_cap, data.p_upper.sum(axis=0))
    per_t = np.minimum(per_t, data.station_cap.sum())
    return float(np.sum(per_t[occ > 0]) * data.dt * data.eta.mean())


def run(seeds=(0, 1, 2)) -> pd.DataFrame:
    rows = []
    for n_evs, n_st in SCALABILITY_CASES:
        for name in EXP5_SCENARIOS:
            for seed in seeds:
                data, evs = scenario_instance(name, n_evs, n_st, seed)
                summ = fleet_summary(evs, data.dt)
                supply = supply_bound_kwh(data)
                rows.append({
                    "n_evs": n_evs, "n_stations": n_st, "scenario": name, "seed": seed,
                    "pct_individually_feasible": summ["pct_individually_feasible"],
                    "total_required_kwh": summ["total_required_kwh"],
                    "supply_bound_kwh": supply,
                    "demand_supply_ratio": summ["total_required_kwh"] / supply if supply > 0 else np.inf,
                    "peak_occupancy": int(occupancy(data.arrival, data.departure, data.T).max()),
                })
    df = pd.DataFrame(rows)
    out = results_dir("scenarios") / "ev_feasibility.csv"
    df.to_csv(out, index=False)
    print(df.groupby(["n_evs", "scenario"])[
        ["pct_individually_feasible", "demand_supply_ratio", "peak_occupancy"]].mean().round(2))
    print(f"\nsaved -> {out}")
    return df


if __name__ == "__main__":
    run()
