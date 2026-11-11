"""
Penalty (rho / c) sweep and fixed-vs-adaptive comparison.

Used to (1) calibrate the automatic penalty of both ADMM variants and
(2) produce the fixed-vs-adaptive table for Experiment 1.

For every run it records
    iterations           until the residual-based stopping rule fires (or max_iter)
    it_gap_1e-3/1e-4     first iteration with optimality gap <= 0.1% / 0.01%
                         and max constraint violation <= 0.1 kW

Run:   python -m experiments.compare_fixed_adaptive [topology] [all|shared|consensus]
       e.g. python -m experiments.compare_fixed_adaptive ring consensus
Out:   results/comparisons/rho_sweep_<topology>_<which>.csv
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from algorithms.base import auto_rho, resolve_rho, ADMMSettings
from algorithms.consensus_admm.consensus_admm import run_consensus_admm
from algorithms.consensus_admm.consensus_utils import auto_c
from algorithms.shared_admm.shared_admm import run_shared_admm
from centralized.solver import solve_centralized
from common.objective import WEIGHT_CASES
from common.utils import results_dir
from ev.fleet import scenario_instance
from network.integration import make_network

SHARED_FACTORS = (0.3, 1.0, 3.0)              # multiples of the (calibrated) automatic rho
CONSENSUS_FACTORS = (3.0, 10.0, 30.0, 100.0, 300.0)
MUS = (10.0, 3.0)                             # residual-balancing thresholds for adaptive runs
CONSENSUS_ADAPT_START = 10.0                  # adaptive consensus starts from this factor
MAX_ITER = 3000
VIOL_TOL = 0.1                                # kW


def iters_to_gap(res, eps: float) -> float:
    h = res.history
    for k, g, v in zip(h["iteration"], h["gap"], h["max_violation"]):
        if g <= eps and v <= VIOL_TOL:
            return int(k) + 1
    return np.nan


def row(algo, topo, factor, adaptive, mu, res, J_star):
    return {
        "algorithm": algo, "topology": topo, "rho_factor": factor,
        "adaptive": adaptive, "mu": mu,
        "iterations": res.iterations, "converged": res.converged,
        "gap_final": abs(res.objective - J_star) / abs(J_star),
        "it_gap_1e-3": iters_to_gap(res, 1e-3),
        "it_gap_1e-4": iters_to_gap(res, 1e-4),
        "viol_final_kw": res.metrics["max_violation"],
        "runtime_s": res.runtime_s, "messages": res.messages,
        "rho_start": res.meta["rho_initial"], "rho_end": res.meta["rho_final"],
        "rho_changes": res.meta["n_rho_changes"],
    }


def run(topology: str = "random", which: str = "all", n_evs: int = 20,
        n_stations: int = 2, seed: int = 0) -> pd.DataFrame:
    data, _ = scenario_instance("morning_peak", n_evs, n_stations, seed)
    w = WEIGHT_CASES["baseline"]
    J_star = solve_centralized(data, w).objective
    g = make_network(topology, data.N, seed=seed)

    rho_shared = resolve_rho(ADMMSettings(), data, w)                         # calibrated auto
    c_cons = auto_c(g.degrees.astype(float), data.N) / auto_rho(data, w)       # original units
    rows = []

    def shared(f, adaptive, mu):
        r = run_shared_admm(data, w, adaptive=adaptive, rho=f * rho_shared, mu=mu,
                            max_iter=MAX_ITER, J_star=J_star)
        rows.append(row("shared_admm", "hub", f, adaptive, mu, r, J_star))
        print(_fmt(rows[-1]), flush=True)

    def consensus(f, adaptive, mu):
        r = run_consensus_admm(data, w, graph=g, adaptive=adaptive, rho=f * c_cons, mu=mu,
                               max_iter=MAX_ITER, J_star=J_star)
        rows.append(row("consensus_admm", topology, f, adaptive, mu, r, J_star))
        print(_fmt(rows[-1]), flush=True)

    print(f"J* = {J_star:.6f}  N={data.N}  topology={topology}  which={which}\n")
    if which in ("all", "shared"):
        for f in SHARED_FACTORS:
            shared(f, False, 3.0)
        for mu in MUS:
            shared(1.0, True, mu)
    if which in ("all", "consensus"):
        for f in CONSENSUS_FACTORS:
            consensus(f, False, 3.0)
        for mu in MUS:
            consensus(CONSENSUS_ADAPT_START, True, mu)

    df = pd.DataFrame(rows)
    out = results_dir("comparisons") / f"rho_sweep_{topology}_{which}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved -> {out}")
    return df


def _fmt(r: dict) -> str:
    def it(x):
        return "  -  " if np.isnan(x) else f"{int(x):5d}"
    return (f"{r['algorithm']:15s} {'adapt' if r['adaptive'] else 'fixed'} mu={r['mu']:4.1f} "
            f"x{r['rho_factor']:<5g} iters={r['iterations']:5d} conv={r['converged']!s:5s} "
            f"to1e-3={it(r['it_gap_1e-3'])} to1e-4={it(r['it_gap_1e-4'])} "
            f"gap={r['gap_final']:.1e} t={r['runtime_s']:5.1f}s "
            f"rho {r['rho_start']:.1e}->{r['rho_end']:.1e} ({r['rho_changes']} changes)")


if __name__ == "__main__":
    topo = sys.argv[1] if len(sys.argv) > 1 else "random"
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    run(topo, which)