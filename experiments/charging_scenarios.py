"""
Experiment 5: charging scenarios.

    morning_peak, evening_peak, random arrivals, high demand, low demand
    (definitions in common/constants.py EXP5_SCENARIOS)

Compares uncoordinated, centralized and both distributed algorithms
(baseline weights).

Run:   python -m experiments.charging_scenarios                   (~4 min / seed)
       python -m experiments.charging_scenarios --seeds 0 1 2
Out:   results/scenarios/exp5_runs.csv, exp5_summary.csv
       plots/scenarios/exp5_metrics.png, exp5_load_profiles.png
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.constants import EXP5_SCENARIOS
from common.objective import WEIGHT_CASES
from common.utils import plots_dir, results_dir
from evaluation.objective_evaluation import DEFAULT_RUNS, label, run_suite
from evaluation.visualization import plot_load_grid, plot_metric_panels
from ev.fleet import scenario_instance

METRICS = ["peak_load_kw", "total_cost_inr", "avg_delay_h", "target_soc_pct",
           "jain_fairness", "total_unmet_kwh", "objective", "gap", "max_violation"]


def run(n_evs: int = 30, n_stations: int = 3, seeds=(0,), max_iter: int = 3000,
        runs=DEFAULT_RUNS, scenarios=tuple(EXP5_SCENARIOS), tag: str = "") -> pd.DataFrame:
    w = WEIGHT_CASES["baseline"]
    rows, panels, d0 = [], {}, None
    for seed in seeds:
        for name in scenarios:
            data, _ = scenario_instance(name, n_evs, n_stations, seed)
            print(f"\nseed {seed}  scenario {name}")
            results, r_rows, _ = run_suite(data, w, runs, name, seed, max_iter)
            rows += r_rows
            if seed == seeds[0]:
                panels[name] = {label(*k): r.P.sum(axis=0) for k, r in results.items()
                                if k[0] != "uncoordinated_fcfs"}
                d0 = data

    df = pd.DataFrame(rows)
    out = results_dir("scenarios")
    df.to_csv(out / f"exp5{tag}_runs.csv", index=False)
    summary = df.groupby(["scenario", "label"], sort=False)[METRICS].mean()
    summary.to_csv(out / f"exp5{tag}_summary.csv")
    pd.set_option("display.width", 220)
    print("\nmean over seeds:\n", summary.round(4).to_string())

    pdir = plots_dir("scenarios")
    plot_metric_panels(df, "scenario", "label", METRICS, pdir / f"exp5{tag}_metrics.png")
    plot_load_grid(panels, d0.grid_cap, pdir / f"exp5{tag}_load_profiles.png",
                   d0.dt, d0.start_hour, ncols=3)
    print(f"\nsaved -> {out}  and  {pdir}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment 5: charging scenarios")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-iter", type=int, default=3000)
    a = p.parse_args()
    run(a.n_evs, a.stations, tuple(a.seeds), a.max_iter)