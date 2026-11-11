"""
Experiment 3: scalability.

    (EVs, stations) = (10,1) (25,2) (50,3) (100,5) (200,10)

Compares execution time, iterations and communication overhead.
Default algorithm/topology pairs (best set-ups from Experiment 1):
    shared_admm (star), shared_admm_adaptive (star), consensus_admm (random)
Shared ADMM on a ring at N=200 needs ~40,000 averaging rounds per iteration,
so ring/mesh are not in the default set (add them with --pairs if needed).

Convergence speed is measured with the load-profile distance
||L - L*|| / ||L*|| (it_dist_*), because at large N the relative objective gap
is dominated by the unavoidable unmet-energy penalty and reaches 0.1% after
one iteration.

Run:   python -m experiments.scalability                       (~5 min)
       python -m experiments.scalability --max-iter 1000       (faster)
       python -m experiments.scalability --seeds 0 1 2         (report version)
       python -m experiments.scalability --pairs shared_admm:star consensus_admm:ring
Out:   results/scalability/exp3_runs.csv, exp3_summary.csv
       plots/scalability/exp3_scalability.png
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.constants import SCALABILITY_CASES
from common.objective import WEIGHT_CASES
from common.utils import plots_dir, results_dir
from evaluation.objective_evaluation import run_suite
from evaluation.visualization import plot_lines
from ev.fleet import scenario_instance

PAIRS = (("shared_admm", "star"), ("shared_admm_adaptive", "star"), ("consensus_admm_adaptive", "random"))
METRICS = ["runtime_s", "iterations", "it_gap_1e-3", "it_dist_1e-2", "it_dist_1e-3", "messages",
           "MB_total", "msgs_per_agent_per_iter", "gap", "final_load_dist", "max_violation"]


def run(cases=SCALABILITY_CASES, seeds=(0,), scenario: str = "random",
        max_iter: int = 2000, pairs=PAIRS, tag: str = "") -> pd.DataFrame:
    w = WEIGHT_CASES["baseline"]
    rows = []
    for n_evs, n_st in cases:
        for seed in seeds:
            data, _ = scenario_instance(scenario, n_evs, n_st, seed)
            print(f"\nN={n_evs} M={n_st} seed={seed}")
            _, r_rows, _ = run_suite(data, w, (("centralized", None),) + tuple(pairs),
                                     scenario, seed, max_iter)
            for r in r_rows:
                it = max(r["iterations"], 1)
                r["msgs_per_agent_per_iter"] = r["messages"] / it / n_evs
                for key in ("MB_total",):
                    r.setdefault(key, 0.0)
                for key in ("it_gap_1e-3", "it_gap_1e-4", "it_dist_1e-2", "it_dist_1e-3",
                            "final_load_dist"):
                    r.setdefault(key, np.nan)
            rows += r_rows

    df = pd.DataFrame(rows)
    out = results_dir("scalability")
    df.to_csv(out / f"exp3{tag}_runs.csv", index=False)
    summary = df.groupby(["n_evs", "label"], sort=False)[METRICS].mean()
    summary.to_csv(out / f"exp3{tag}_summary.csv")
    pd.set_option("display.width", 220)
    print("\nmean over seeds:\n", summary.round(4).to_string())

    pdir = plots_dir("scalability")
    plot_lines(df, "n_evs", "label",
               ["runtime_s", "iterations", "it_dist_1e-2", "MB_total", "msgs_per_agent_per_iter",
                "final_load_dist"],
               pdir / f"exp3{tag}_scalability.png", logy=("runtime_s", "MB_total", "final_load_dist"))
    print(f"\nsaved -> {out}  and  {pdir}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment 3: scalability")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--scenario", default="random")
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--pairs", nargs="+", default=None,
                   help="algorithm:topology pairs, e.g. shared_admm:star consensus_admm:ring")
    a = p.parse_args()
    pairs = PAIRS if a.pairs is None else tuple(tuple(x.split(":")) for x in a.pairs)
    run(seeds=tuple(a.seeds), scenario=a.scenario, max_iter=a.max_iter, pairs=pairs)