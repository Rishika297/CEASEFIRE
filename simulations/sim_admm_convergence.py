"""
Animation: distributed agents converging to the centralized optimum.
Left: Shared ADMM (adaptive, star).  Right: Consensus ADMM (adaptive, random graph).
Top row: total load of the current iterate vs the optimum.  Bottom: optimality gap.

Run:  python -m simulations.sim_admm_convergence
      python -m simulations.sim_admm_convergence --grid-factor 0.35   (congested grid)
Out:  plots/animations/admm_convergence[_gridX].gif / .png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from common.objective import WEIGHT_CASES
from network.integration import run_algorithm
from simulations.sim_utils import COLORS, clock_ticks, instance, iterates, png_path, save_gif

MAX_FRAMES = 90


def main(scenario: str = "morning_peak", n_evs: int = 30, n_stations: int = 3, seed: int = 0,
         grid_factor=None, case: str = "baseline"):
    data = instance(scenario, n_evs, n_stations, seed, grid_factor)
    w = WEIGHT_CASES[case]
    cen = run_algorithm("centralized", data, w)
    L_star = cen.P.sum(0)
    runs = {
        "Shared ADMM (adaptive, star)": (run_algorithm(
            "shared_admm_adaptive", data, w, topology="star", J_star=cen.objective,
            store_iterates=True, store_every=2), COLORS["shared"], 2),
        "Consensus ADMM (adaptive, random)": (run_algorithm(
            "consensus_admm_adaptive", data, w, topology="random", J_star=cen.objective,
            store_iterates=True, store_every=20), COLORS["consensus"], 20),
    }
    its = {k: iterates(r) for k, (r, _, _) in runs.items()}
    n = max(len(v) for v in its.values())
    stride = max(1, int(np.ceil(n / MAX_FRAMES)))
    frames = list(range(0, n, stride))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    T, x = data.T, np.arange(data.T)
    ymax = 1.15 * max(data.grid_cap.max(), L_star.max(),
                      max(it[0].sum(0).max() if len(it) else 0 for it in its.values()))
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5), gridspec_kw={"height_ratios": [3, 2]})
    artists = {}
    for col, (name, (r, color, every)) in enumerate(runs.items()):
        ax, axg = axes[0, col], axes[1, col]
        ax.plot(x, data.grid_cap, "k--", lw=1, label="grid capacity")
        ax.step(x, L_star, where="post", color=COLORS["centralized"], lw=2.5, alpha=0.5,
                label="centralized optimum")
        line = ax.step(x, np.zeros(T), where="post", color=color, lw=1.8, label="ADMM iterate")[0]
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, T - 1)
        ax.set_title(name, fontsize=10)
        clock_ticks(ax, T, data.dt, data.start_hour)
        ax.legend(fontsize=7, loc="upper right")
        h = r.history
        gap = np.maximum(np.asarray(h["gap"]), 1e-8)
        axg.plot(h["iteration"], gap, color="0.85")
        prog = axg.plot([], [], color=color, lw=1.8)[0]
        dot = axg.plot([], [], "o", color=color)[0]
        axg.set_yscale("log")
        axg.set_xlabel("iteration")
        axg.set_ylabel("optimality gap")
        txt = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", fontsize=8,
                      bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))
        artists[name] = (line, prog, dot, txt, gap, h, every)
    axes[0, 0].set_ylabel("total load [kW]")
    fig.suptitle("Distributed EV agents converging to the centralized optimum", fontsize=11)

    def update(fi):
        j = frames[fi]
        for name, (line, prog, dot, txt, gap, h, every) in artists.items():
            it = its[name]
            jj = min(j, len(it) - 1)
            line.set_ydata(it[jj].sum(0))
            k = min(jj * every, len(gap) - 1)
            prog.set_data(h["iteration"][:k + 1], gap[:k + 1])
            dot.set_data([h["iteration"][k]], [gap[k]])
            done = " (converged)" if jj == len(it) - 1 else ""
            txt.set_text(f"iteration {int(h['iteration'][k])}{done}\n"
                         f"gap {gap[k]:.1e}   violation {h['max_violation'][k]:.2f} kW")

    update(len(frames) - 1)
    tag = "" if grid_factor is None else f"_grid{grid_factor:g}"
    fig.tight_layout()
    fig.savefig(png_path(f"admm_convergence{tag}"), dpi=150)
    save_gif(fig, update, len(frames), f"admm_convergence{tag}", fps=8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-factor", type=float, default=None)
    p.add_argument("--case", default="baseline", choices=list(WEIGHT_CASES))
    a = p.parse_args()
    main(a.scenario, a.n_evs, a.stations, a.seed, a.grid_factor, a.case)