"""
Experiment 2: objective-weight sensitivity.

    baseline (1,1,1), peak priority (10,1,1), delay priority (1,10,1),
    fairness priority (1,1,10)   -- (alpha, beta, gamma)

Same weights for centralized and distributed algorithms in every case.
Reports objective components, physical metrics and charging schedules.
(Sensitivity experiment, not a convergence experiment.)

The fairness weight only has an effect when grid capacity is scarce (otherwise
every EV gets what it can physically take). Use --grid-factor to make the grid
tighter, e.g. 0.35 (default factor is 0.7 x total station capacity).

Run:   python -m experiments.objective_weight_sensitivity                     (~3 min / seed)
       python -m experiments.objective_weight_sensitivity --grid-factor 0.35
       python -m experiments.objective_weight_sensitivity --seeds 0 1 2
Out:   results/sensitivity/exp2[_gridX]_runs.csv, exp2[_gridX]_summary.csv
       plots/objective_sensitivity/exp2[_gridX]_components.png, _metrics.png,
       _load_profiles.png, _schedules_centralized.png
"""
from __future__ import annotations

import argparse
from typing import Optional

import numpy as np
import pandas as pd

from common.objective import WEIGHT_CASES
from common.utils import plots_dir, results_dir
from evaluation.objective_evaluation import DEFAULT_RUNS, label, run_suite
from evaluation.visualization import plot_load_grid, plot_metric_panels, plot_schedule_heatmaps
from ev.fleet import scenario_instance

COMPONENTS = ["J_cost", "J_peak", "J_delay", "J_fair", "objective"]
PHYSICAL = ["peak_load_kw", "total_cost_inr", "avg_delay_h", "target_soc_pct",
            "jain_fairness", "total_unmet_kwh"]
# Experiment 2 compares only methods that actually optimise J with the given weights
EXP2_RUNS = (
    ("centralized", None),
    ("shared_admm_adaptive", "star"),
    ("consensus_admm_adaptive", "random"),
)

def run(n_evs: int = 30, n_stations: int = 3, scenario: str = "morning_peak", seeds=(0,),
        max_iter: int = 3000, runs=EXP2_RUNS, tag: str = "",
        grid_factor: Optional[float] = None) -> pd.DataFrame:
    if grid_factor is not None and not tag:
        tag = f"_grid{grid_factor:g}"
    extra = {} if grid_factor is None else {"grid_cap_factor": grid_factor}

    rows, first = [], {}
    for seed in seeds:
        data, _ = scenario_instance(scenario, n_evs, n_stations, seed, **extra)
        for case, w in WEIGHT_CASES.items():
            print(f"\nseed {seed}  case {case}  (alpha={w.alpha}, beta={w.beta}, gamma={w.gamma})"
                  f"  grid cap = {data.grid_cap.max():.1f} kW")
            results, r_rows, _ = run_suite(data, w, runs, scenario, seed, max_iter)
            for r in r_rows:
                r.update(case=case, alpha=w.alpha, beta=w.beta, gamma=w.gamma,
                         grid_factor=grid_factor if grid_factor is not None else 0.7)
            rows += r_rows
            if seed == seeds[0]:
                first[case] = (data, results)

    df = pd.DataFrame(rows)
    out = results_dir("sensitivity")
    df.to_csv(out / f"exp2{tag}_runs.csv", index=False)
    summary = df.groupby(["case", "label"], sort=False)[COMPONENTS + ["gap"] + PHYSICAL].mean()
    summary.to_csv(out / f"exp2{tag}_summary.csv")
    pd.set_option("display.width", 220)
    print("\nmean over seeds:\n", summary.round(4).to_string())

    pdir = plots_dir("objective_sensitivity")
    plot_metric_panels(df, "case", "label", COMPONENTS, pdir / f"exp2{tag}_components.png")
    plot_metric_panels(df, "case", "label", PHYSICAL, pdir / f"exp2{tag}_metrics.png")

    show = [k for k in runs if k[0] in ("uncoordinated_asap", "centralized")
            or k[0].startswith(("shared", "consensus"))]
    panels = {case: {label(*k): res[k].P.sum(axis=0) for k in show}
              for case, (d, res) in first.items()}
    d0 = next(iter(first.values()))[0]
    plot_load_grid(panels, d0.grid_cap, pdir / f"exp2{tag}_load_profiles.png",
                   d0.dt, d0.start_hour, ncols=2)
    order = np.argsort(d0.arrival, kind="stable")
    plot_schedule_heatmaps({case: res[("centralized", None)].P for case, (d, res) in first.items()},
                           pdir / f"exp2{tag}_schedules_centralized.png",
                           d0.dt, d0.start_hour, order=order)
    print(f"\nsaved -> {out}  and  {pdir}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment 2: objective-weight sensitivity")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--grid-factor", type=float, default=None,
                   help="grid capacity = factor x total station capacity (default 0.7)")
    a = p.parse_args()
    run(a.n_evs, a.stations, a.scenario, tuple(a.seeds), a.max_iter,
        grid_factor=a.grid_factor)