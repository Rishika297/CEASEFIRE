"""
Compare ADMM variants on ONE instance and print convergence traces.

Prints gap, load distance ||L - L*||/||L*||, max constraint violation, the
largest network-average price, and how much the powers (dX) and load
estimates (dW) changed, at checkpoint iterations; plots the full curves.

Run (example: the case where consensus ADMM struggled):
    python -m experiments.compare_algorithms --case fairness_priority --grid-factor 0.35 \
        --max-iter 6000 --consensus-factors 1 --consensus-adaptive --no-shared
Options:
    --scenario morning_peak --n-evs 30 --stations 3 --seed 0 --topology random
    --consensus-factors   multiples of the default consensus penalty to try (fixed c)
    --consensus-adaptive  also run consensus_admm_adaptive from the default c
    --no-shared           skip Shared ADMM
Out:  plots/convergence/compare_<case>_grid<g>.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithms.base import auto_rho
from algorithms.consensus_admm.consensus_utils import auto_c
from centralized.solver import solve_centralized
from common.objective import WEIGHT_CASES
from common.utils import plots_dir
from ev.fleet import scenario_instance
from network.integration import make_network, run_algorithm

CHECKPOINTS = (50, 100, 300, 1000, 2000, 3000, 4000, 6000, 8000, 10000)


def _get(h, key, i):
    return h[key][i] if key in h and i < len(h[key]) else float("nan")


def trace_line(name, r):
    h = r.history
    print(f"\n{name}  (iters={r.iterations}, converged={r.converged}, "
          f"rho {r.meta['rho_initial']:.2e} -> {r.meta['rho_final']:.2e}, t={r.runtime_s:.1f}s)")
    for k in CHECKPOINTS:
        if k <= len(h["iteration"]):
            i = k - 1
            print(f"    k={k:5d}: gap={_get(h, 'gap', i):.1e} dist={_get(h, 'load_dist', i):.1e} "
                  f"viol={_get(h, 'max_violation', i):6.2f}  price_max={_get(h, 'price_max', i):.2e} "
                  f"dX={_get(h, 'r_x', i):.1e} dW={_get(h, 'r_w', i):.1e} rho={_get(h, 'rho', i):.1e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--case", default="baseline", choices=list(WEIGHT_CASES))
    p.add_argument("--grid-factor", type=float, default=None)
    p.add_argument("--topology", default="random")
    p.add_argument("--max-iter", type=int, default=6000)
    p.add_argument("--consensus-factors", type=float, nargs="*", default=[1.0])
    p.add_argument("--consensus-adaptive", action="store_true",
                   help="also run consensus_admm_adaptive (starts from the default c)")
    p.add_argument("--no-shared", action="store_true")
    a = p.parse_args()

    extra = {} if a.grid_factor is None else {"grid_cap_factor": a.grid_factor}
    data, _ = scenario_instance(a.scenario, a.n_evs, a.stations, a.seed, **extra)
    w = WEIGHT_CASES[a.case]
    cen = solve_centralized(data, w)
    J_star, L_star = cen.objective, cen.P.sum(axis=0)
    g = make_network(a.topology, data.N, seed=a.seed)
    c_default = auto_c(g.degrees.astype(float), data.N) / auto_rho(data, w)
    print(f"case={a.case} grid cap={data.grid_cap.max():.1f} kW  J*={J_star:.5f}  "
          f"default consensus c={c_default:.2e}")

    runs = {}
    if not a.no_shared:
        runs["shared_admm_adaptive (star)"] = run_algorithm(
            "shared_admm_adaptive", data, w, topology="star", J_star=J_star, L_star=L_star,
            max_iter=a.max_iter)
    for f in a.consensus_factors:
        runs[f"consensus_admm ({a.topology}) c x{f:g}"] = run_algorithm(
            "consensus_admm", data, w, graph=g, J_star=J_star, L_star=L_star,
            max_iter=a.max_iter, rho=f * c_default)
    if a.consensus_adaptive:
        runs[f"consensus_admm_adaptive ({a.topology})"] = run_algorithm(
            "consensus_admm_adaptive", data, w, graph=g, J_star=J_star, L_star=L_star,
            max_iter=a.max_iter)
    for name, r in runs.items():
        trace_line(name, r)

    panels = [("gap", "optimality gap"), ("load_dist", "||L - L*|| / ||L*||"),
              ("max_violation", "max violation [kW]"), ("rho", "penalty rho / c")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 3.8))
    for name, r in runs.items():
        h = r.history
        for ax, (key, _) in zip(axes, panels):
            if key in h:
                ax.plot(h["iteration"], np.maximum(h[key], 1e-12), label=name)
    for ax, (_, title) in zip(axes, panels):
        ax.set_yscale("log"); ax.set_title(title); ax.set_xlabel("iteration"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    gtag = "default" if a.grid_factor is None else f"{a.grid_factor:g}"
    out = plots_dir("convergence") / f"compare_{a.case}_grid{gtag}.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()