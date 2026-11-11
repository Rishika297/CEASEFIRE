"""
Animation: total charging load through the day.
Uncoordinated vs centralized vs Shared ADMM vs Consensus ADMM (adaptive).

Run:  python -m simulations.sim_grid_load
      python -m simulations.sim_grid_load --scenario evening_peak
Out:  plots/animations/grid_load_<scenario>.gif / .png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from common.objective import WEIGHT_CASES
from network.integration import run_algorithm
from simulations.sim_utils import COLORS, clock_ticks, instance, png_path, save_gif


def main(scenario: str = "morning_peak", n_evs: int = 30, n_stations: int = 3, seed: int = 0):
    data = instance(scenario, n_evs, n_stations, seed)
    w = WEIGHT_CASES["baseline"]
    asap = run_algorithm("uncoordinated_asap", data, w)
    cen = run_algorithm("centralized", data, w)
    sh = run_algorithm("shared_admm_adaptive", data, w, topology="star", J_star=cen.objective)
    co = run_algorithm("consensus_admm_adaptive", data, w, topology="random", J_star=cen.objective)

    curves = {
        "Uncoordinated (charge on arrival)": (asap.P.sum(0), COLORS["uncoordinated"], "-", 2.4),
        "Centralized optimum": (cen.P.sum(0), COLORS["centralized"], "-", 2.4),
        "Shared ADMM (star)": (sh.P.sum(0), COLORS["shared"], "--", 1.6),
        "Consensus ADMM (random graph)": (co.P.sum(0), COLORS["consensus"], ":", 1.8),
    }
    T, x = data.T, np.arange(data.T)
    ymax = 1.1 * max(max(L.max() for L, *_ in curves.values()), data.grid_cap.max())

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x, data.grid_cap, "k--", lw=1.2, label="grid capacity")
    lines = {}
    for name, (L, col, ls, lw) in curves.items():
        lines[name] = ax.step([], [], where="post", color=col, ls=ls, lw=lw, label=name)[0]
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("total charging load [kW]")
    ax.set_title(f"EV charging load - {scenario.replace('_', ' ')} ({data.N} EVs, {data.M} stations)")
    ax.legend(loc="upper right", fontsize=8)
    info = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", family="monospace",
                   fontsize=8, bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))
    v1 = ax.axvline(0, color="k", lw=0.8, alpha=0.4)
    ax2.step(x, data.prices, where="post", color="0.4")
    ax2.set_ylabel("price\n[Rs/kWh]")
    v2 = ax2.axvline(0, color="k", lw=0.8, alpha=0.4)
    clock_ticks(ax2, T, data.dt, data.start_hour)

    def update(t):
        for name, (L, *_ ) in curves.items():
            lines[name].set_data(x[:t + 1], L[:t + 1])
        v1.set_xdata([t, t])
        v2.set_xdata([t, t])
        rows = []
        for name, (L, *_ ) in curves.items():
            cost = float(data.prices[:t + 1] @ L[:t + 1] * data.dt)
            rows.append(f"{name.split(' (')[0]:22s} peak {L[:t + 1].max():6.1f} kW   cost Rs {cost:7.0f}")
        info.set_text("\n".join(rows))

    update(T - 1)
    fig.savefig(png_path(f"grid_load_{scenario}"), dpi=150)
    save_gif(fig, update, T, f"grid_load_{scenario}", fps=6)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.scenario, a.n_evs, a.stations, a.seed)