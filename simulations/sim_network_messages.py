"""
Animation: the communication graph during Consensus ADMM (adaptive).
Nodes = EVs (circle layout), coloured by how far each EV's delivered energy is
from its optimal value (green = at optimum, red = far). Edges pulse to show the
neighbour message exchange. Right: energy per EV, current vs optimal.

Run:  python -m simulations.sim_network_messages
      python -m simulations.sim_network_messages --topology ring --n-evs 16
Out:  plots/animations/network_<topology>.gif / .png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from common.objective import WEIGHT_CASES
from network.integration import make_network, run_algorithm
from simulations.sim_utils import COLORS, instance, iterates, png_path, save_gif

EVERY = 10
MAX_FRAMES = 80


def main(topology: str = "random", n_evs: int = 20, n_stations: int = 2, seed: int = 0):
    data = instance("morning_peak", n_evs, n_stations, seed)
    w = WEIGHT_CASES["baseline"]
    g = make_network(topology, data.N, seed=seed)
    cen = run_algorithm("centralized", data, w)
    r = run_algorithm("consensus_admm_adaptive", data, w, graph=g, J_star=cen.objective,
                      store_iterates=True, store_every=EVERY)
    it = iterates(r)
    k_eta = data.eta * data.dt
    E_star = k_eta * cen.P.sum(1)
    scale = np.maximum(data.R, 1.0)

    ang = 2 * np.pi * np.arange(data.N) / data.N
    pos = np.c_[np.cos(ang), np.sin(ang)]
    segs = [[pos[i], pos[j]] for i, j in g.edges]

    n = len(it)
    stride = max(1, int(np.ceil(n / MAX_FRAMES)))
    frames = list(range(0, n, stride)) + ([n - 1] if (n - 1) % stride else [])

    fig, (ax, axb) = plt.subplots(1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1, 1.3]})
    lc = LineCollection(segs, colors="0.6", linewidths=0.8, alpha=0.3)
    ax.add_collection(lc)
    nodes = ax.scatter(pos[:, 0], pos[:, 1], s=260, c=np.zeros(data.N), cmap="RdYlGn_r",
                       vmin=0, vmax=0.5, edgecolors="k", zorder=3)
    for i in range(data.N):
        ax.text(*(1.13 * pos[i]), str(i), ha="center", va="center", fontsize=7)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(nodes, ax=ax, shrink=0.7, label="|energy - optimal| / required energy")
    title = ax.set_title("")

    idx = np.arange(data.N)
    axb.bar(idx, E_star, color=COLORS["centralized"], alpha=0.35, label="optimal (centralized)")
    cur = axb.bar(idx, np.zeros(data.N), width=0.5, color=COLORS["consensus"], label="current iterate")
    axb.set_ylim(0, 1.15 * max(E_star.max(), (k_eta * np.asarray(it).sum(2)).max()))
    axb.set_xlabel("EV")
    axb.set_ylabel("energy delivered [kWh]")
    axb.legend(fontsize=8)
    fig.suptitle(f"Consensus ADMM on a {topology} graph: each EV only talks to its neighbours",
                 fontsize=11)

    def update(fi):
        j = frames[fi]
        P = it[j]
        E = k_eta * P.sum(1)
        nodes.set_array(np.abs(E - E_star) / scale)
        lc.set_alpha(0.75 if fi % 2 == 0 else 0.25)
        for b, e in zip(cur, E):
            b.set_height(e)
        h = r.history
        k = min(j * EVERY, len(h["gap"]) - 1)
        title.set_text(f"iteration {int(h['iteration'][k])}   gap {h['gap'][k]:.1e}   "
                       f"messages {int(h['messages'][k]):,}")
    update(len(frames) - 1)
    fig.savefig(png_path(f"network_{topology}"), dpi=150)
    save_gif(fig, update, len(frames), f"network_{topology}", fps=8)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topology", default="random", choices=["ring", "star", "mesh", "random"])
    p.add_argument("--n-evs", type=int, default=20)
    p.add_argument("--stations", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.topology, a.n_evs, a.stations, a.seed)