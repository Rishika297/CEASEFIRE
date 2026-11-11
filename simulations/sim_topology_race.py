"""
Animation: Consensus ADMM (adaptive) on ring / star / mesh / random, side by side.
Each panel: load of the current iterate vs the optimum, iteration, gap and data sent.

Run:  python -m simulations.sim_topology_race
Out:  plots/animations/topology_race.gif / .png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from common.objective import WEIGHT_CASES
from network.integration import run_algorithm
from simulations.sim_utils import COLORS, TOPO_COLORS, instance, iterates, png_path, save_gif

EVERY = 20
MAX_FRAMES = 90


def main(scenario: str = "morning_peak", n_evs: int = 30, n_stations: int = 3, seed: int = 0):
    data = instance(scenario, n_evs, n_stations, seed)
    w = WEIGHT_CASES["baseline"]
    cen = run_algorithm("centralized", data, w)
    L_star = cen.P.sum(0)
    runs = {}
    for topo in ("ring", "star", "mesh", "random"):
        print(f"running consensus ADMM on {topo} ...", flush=True)
        r = run_algorithm("consensus_admm_adaptive", data, w, topology=topo, graph_seed=seed,
                          J_star=cen.objective, store_iterates=True, store_every=EVERY)
        runs[topo] = (r, iterates(r))
    n = max(len(it) for _, it in runs.values())
    stride = max(1, int(np.ceil(n / MAX_FRAMES)))
    frames = list(range(0, n, stride)) + ([n - 1] if (n - 1) % stride else [])

    T, x = data.T, np.arange(T := data.T)
    ymax = 1.15 * max(data.grid_cap.max(), L_star.max())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    art = {}
    floats_per_msg = data.M * data.T
    for ax, (topo, (r, it)) in zip(axes.flat, runs.items()):
        ax.plot(x, data.grid_cap, "k--", lw=1)
        ax.step(x, L_star, where="post", color=COLORS["centralized"], lw=2.5, alpha=0.45)
        line = ax.step(x, np.zeros(T), where="post", color=TOPO_COLORS[topo], lw=1.8)[0]
        ax.set_ylim(0, ymax)
        ax.set_title(topo, fontsize=11, color=TOPO_COLORS[topo])
        txt = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top", fontsize=8,
                      family="monospace", bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))
        art[topo] = (line, txt)
    fig.suptitle("Consensus ADMM (adaptive) on different communication topologies\n"
                 "faded blue = centralized optimum, dashed = grid capacity", fontsize=10)

    def update(fi):
        j = frames[fi]
        for topo, (r, it) in runs.items():
            line, txt = art[topo]
            jj = min(j, len(it) - 1)
            line.set_ydata(it[jj].sum(0))
            h = r.history
            k = min(jj * EVERY, len(h["gap"]) - 1)
            mb = h["messages"][k] * floats_per_msg * 8 / 1e6
            done = "  CONVERGED" if jj == len(it) - 1 else ""
            txt.set_text(
    f"iter {int(h['iteration'][k]):5d} {done}\n"
    f"gap {h['gap'][k]:.1e}\n"
    f"data {mb:8.1f} MB"
)


    update(len(frames) - 1)
    fig.tight_layout()
    fig.savefig(png_path("topology_race"), dpi=150)
    save_gif(fig, update, len(frames), "topology_race", fps=8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.scenario, a.n_evs, a.stations, a.seed)