"""
STL verification benchmark: checks every algorithm's schedule against

    Req 1  F_[a_i, d_i] (SoC_i >= SoC_i^target)   target SoC before departure
    Req 2  G_[0, T-1]  (P_total <= P_grid)         grid capacity never exceeded

on identical instances (uncoordinated ASAP/FCFS, centralized, Shared ADMM,
Consensus ADMM), and reports satisfaction + robustness side by side.

Run:   python -m experiments.stl_verification                       (5 scenarios, ~4 min)
       python -m experiments.stl_verification --scenarios morning_peak --grid-factor 0.5
       python -m experiments.stl_verification --no-admm              (fast, no ADMM)
Out:   results/verification/stl<tag>_runs.csv        one row per (scenario, seed, algorithm)
       results/verification/stl<tag>_summary.csv     mean over seeds
       results/verification/stl<tag>_evs.csv         per-EV verdicts
       results/verification/reports/<scenario>_seed<k>_<algorithm>.txt
       plots/verification/stl<tag>_comparison.png
       plots/verification/<scenario>_<algorithm>_{grid,soc}.png  (first seed)
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.constants import EXP5_SCENARIOS
from common.objective import WEIGHT_CASES
from common.utils import plots_dir, results_dir
from evaluation.objective_evaluation import DEFAULT_RUNS, label, run_row, run_suite
from ev.fleet import scenario_instance
from verification.stl_monitor import verify_result
from verification.verification_report import (plot_grid_verification,
                                              plot_soc_verification,
                                              plot_stl_comparison)

SUMMARY_COLS = ["stl_satisfied", "stl_satisfied_reachable", "stl_req1_sat_pct",
                "stl_req1_reachable_sat_pct", "stl_req1_n_violated", "stl_req1_n_unreachable", "stl_req1_rho_min", "stl_req1_min_margin_h",
                "stl_req1_mean_margin_h", "stl_req2_satisfied", "stl_req2_rho_kw",
                "stl_req2_rho_norm", "stl_req2_viol_hours", "stl_req2_overload_kwh",
                "stl_rho_overall", "peak_load_kw", "total_cost_inr", "target_soc_pct"]


def _slug(s: str) -> str:
    return s.replace(" ", "_").replace("(", "").replace(")", "")


def run(scenarios=tuple(EXP5_SCENARIOS), n_evs: int = 30, n_stations: int = 3,
        seeds=(0,), max_iter: int = 3000, grid_factor=None, case: str = "baseline",
        runs=DEFAULT_RUNS, tag: str = "", plots: bool = True) -> pd.DataFrame:
    w = WEIGHT_CASES[case]
    extra = {} if grid_factor is None else {"grid_cap_factor": grid_factor}
    out = results_dir("verification")
    rep_dir = results_dir("verification/reports")
    pdir = plots_dir("verification")
    rows, ev_tables = [], []

    for seed in seeds:
        for name in scenarios:
            data, _ = scenario_instance(name, n_evs, n_stations, seed, **extra)
            print(f"\nseed {seed}  scenario {name}  (grid cap {data.grid_cap.max():.1f} kW)")
            results, _, J_star = run_suite(data, w, runs, name, seed, max_iter, verbose=False)
            for (algo, topo), res in results.items():
                lab = label(algo, topo)
                rep = verify_result(res, data)
                rep.algorithm = lab
                row = run_row(res, J_star, grid_factor=grid_factor, case=case)
                row.update(rep.to_row())
                rows.append(row)

                evt = rep.ev_table()
                evt.insert(0, "label", lab)
                evt.insert(0, "seed", seed)
                evt.insert(0, "scenario", name)
                ev_tables.append(evt)

                (rep_dir / f"{name}_seed{seed}_{_slug(lab)}.txt").write_text(rep.text(max_evs=None))
                if plots and seed == seeds[0]:
                    plot_grid_verification(data, res.P, rep, pdir / f"{name}_{_slug(lab)}_grid.png")
                    plot_soc_verification(data, res.P, rep, pdir / f"{name}_{_slug(lab)}_soc.png")

                r = rep.to_row()
                print(f"    {lab:34s} STL(reachable)={'SAT ' if r['stl_satisfied_reachable'] else 'VIOL'} | "
                      f"Req1 {r['stl_req1_sat_pct']:5.1f}% EVs ok "
                      f"(reachable ok {r['stl_req1_reachable_sat_pct']:5.1f}%, unreachable {r['stl_req1_n_unreachable']}), "
                      f"rho_min={100 * r['stl_req1_rho_min']:+6.2f}pp | "
                      f"Req2 rho={r['stl_req2_rho_kw']:+7.2f}kW, "
                      f"{r['stl_req2_viol_hours']:.1f}h over cap", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"stl{tag}_runs.csv", index=False)
    pd.concat(ev_tables).to_csv(out / f"stl{tag}_evs.csv", index=False)
    summary = df.groupby(["scenario", "label"], sort=False)[SUMMARY_COLS].mean()
    summary.to_csv(out / f"stl{tag}_summary.csv")
    pd.set_option("display.width", 250)
    show = ["stl_satisfied_reachable", "stl_req1_sat_pct", "stl_req1_reachable_sat_pct",
            "stl_req1_mean_margin_h", "stl_req2_rho_kw", "stl_req2_viol_hours"]
    print("\nmean over seeds:\n", summary[show].round(3).to_string())
    if plots:
        plot_stl_comparison(df, pdir / f"stl{tag}_comparison.png")
    print(f"\nsaved -> {out}  and  {pdir}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="STL verification of all algorithms")
    p.add_argument("--scenarios", nargs="+", default=list(EXP5_SCENARIOS))
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--grid-factor", type=float, default=None,
                   help="grid capacity as a fraction of total station capacity (default 0.7)")
    p.add_argument("--case", default="baseline", choices=list(WEIGHT_CASES))
    p.add_argument("--no-admm", action="store_true", help="only uncoordinated + centralized")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    runs = tuple(r for r in DEFAULT_RUNS if not (a.no_admm and "admm" in r[0]))
    run(tuple(a.scenarios), a.n_evs, a.stations, tuple(a.seeds), a.max_iter,
        a.grid_factor, a.case, runs, a.tag, not a.no_plots)
