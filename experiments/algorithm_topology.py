"""
Experiment 1: ADMM algorithms x communication topologies.

    algorithms : shared_admm, shared_admm_adaptive, consensus_admm, consensus_admm_adaptive
    topologies : ring, star, mesh, random
    reference  : centralized optimum J*

Metrics per run: iterations (stopping rule), iterations / messages to 0.1% and
0.01% gap, final primal & dual residuals, runtime, final objective and gap,
messages and MB sent (communication overhead).

Run:   python -m experiments.algorithm_topology                       (1 seed, ~10 min)
       python -m experiments.algorithm_topology --seeds 0 1 2         (report version)
       python -m experiments.algorithm_topology --n-evs 20 --max-iter 1500   (quick check)
Out:   results/topologies/exp1_runs.csv, exp1_summary.csv, <algo>_<topo>_seed<k>.npz
       plots/topology_comparison/exp1_*.png
"""
from __future__ import annotations

import argparse

import pandas as pd

from centralized.solver import solve_centralized
from common.constants import TOPOLOGIES
from common.objective import WEIGHT_CASES
from common.utils import plots_dir, results_dir
from evaluation.convergence import convergence_summary
from evaluation.visualization import plot_convergence_grid, plot_metric_bars
from ev.fleet import scenario_instance
from network.integration import ADMM_ALGORITHMS, run_algorithm

SUMMARY_COLS = ["iterations", "converged", "it_gap_1e-3", "it_gap_1e-4", "it_dist_1e-2",
                "it_dist_1e-3", "msgs_gap_1e-3", "msgs_dist_1e-2", "final_r_primal",
                "final_r_dual", "runtime_s", "objective", "gap_final", "messages",
                "MB_total", "max_violation"]


def run(n_evs: int = 30, n_stations: int = 3, scenario: str = "morning_peak",
        seeds=(0,), max_iter: int = 3000, algorithms=ADMM_ALGORITHMS,
        topologies=TOPOLOGIES, save_runs: bool = True) -> pd.DataFrame:
    w = WEIGHT_CASES["baseline"]
    rows, curves = [], {}
    out_dir = results_dir("topologies")

    for seed in seeds:
        data, _ = scenario_instance(scenario, n_evs, n_stations, seed)
        cen = solve_centralized(data, w, scenario=scenario, seed=seed)
        J_star = cen.objective
        print(f"\nseed {seed}: N={data.N} M={data.M}  J* = {J_star:.6f}  (centralized {cen.runtime_s:.2f}s)")

        for algo in algorithms:
            for topo in topologies:
                r = run_algorithm(algo, data, w, topology=topo, graph_seed=seed,
                                  scenario=scenario, seed=seed, max_iter=max_iter, J_star=J_star,L_star = cen.P.sum(axis=0))
                row = {"algorithm": algo, "topology": topo, "seed": seed, "n_evs": data.N,
                       "iterations": r.iterations, "converged": r.converged,
                       "runtime_s": r.runtime_s, "objective": r.objective,
                       "J_star": J_star, "gap_final": abs(r.objective - J_star) / abs(J_star),
                       "messages": r.messages,
                       "max_violation": r.metrics["max_violation"],
                       "rho_start": r.meta["rho_initial"], "rho_end": r.meta["rho_final"]}
                row.update(convergence_summary(r))
                rows.append(row)
                if seed == seeds[0]:
                    curves[(algo, topo)] = r
                if save_runs:
                    r.save(out_dir / f"{algo}_{topo}_seed{seed}")
                print(f"  {algo:24s} {topo:6s} iters={r.iterations:5d} conv={r.converged!s:5s} "
                      f"to1e-3={row['it_gap_1e-3']:>6} gap={row['gap_final']:.1e} "
                      f"t={r.runtime_s:6.1f}s msgs={r.messages:9d} MB={row['MB_total']:8.1f}",
                      flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "exp1_runs.csv", index=False)
    summary = df.groupby(["algorithm", "topology"])[SUMMARY_COLS].mean(numeric_only=True)
    summary.to_csv(out_dir / "exp1_summary.csv")
    pd.set_option("display.width", 220)
    print("\nmean over seeds:\n", summary.round(4).to_string())

    pdir = plots_dir("topology_comparison")
    plot_convergence_grid(curves, "gap", pdir / "exp1_gap.png", ylabel="optimality gap")
    plot_convergence_grid(curves, "r_primal", pdir / "exp1_primal_residual.png",
                          ylabel="primal residual [kW]")
    plot_convergence_grid(curves, "r_dual", pdir / "exp1_dual_residual.png",
                          ylabel="dual residual [kW]")
    plot_convergence_grid(curves, "load_dist", pdir / "exp1_load_distance.png",
                        ylabel="||L - L*|| / ||L*||")
    plot_metric_bars(df, "it_gap_1e-3", pdir / "exp1_iters_to_gap.png",
                     ylabel="iterations to 0.1% gap")
    plot_metric_bars(df, "MB_total", pdir / "exp1_communication.png", logy=True,
                     ylabel="data sent [MB]")
    plot_metric_bars(df, "runtime_s", pdir / "exp1_runtime.png", ylabel="runtime [s]")
    print(f"\nsaved -> {out_dir}  and  {pdir}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Experiment 1: algorithms x topologies")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--max-iter", type=int, default=3000)
    p.add_argument("--algorithms", nargs="+", default=list(ADMM_ALGORITHMS))
    p.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES))
    a = p.parse_args()
    run(a.n_evs, a.stations, a.scenario, tuple(a.seeds), a.max_iter,
        tuple(a.algorithms), tuple(a.topologies))