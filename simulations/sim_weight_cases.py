"""
Animation: how the objective weights change the optimal behaviour (Experiment 2).
Cycles through baseline / peak / delay / fairness priority. Left: total load
(current case bold, others faint). Right: charging schedule heatmap.

Run:  python -m simulations.sim_weight_cases
      python -m simulations.sim_weight_cases --grid-factor 0.5
Out:  plots/animations/weight_cases[_gridX].gif / .png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from common.objective import WEIGHT_CASES
from network.integration import run_algorithm
from simulations.sim_utils import clock_ticks, instance, png_path, save_gif

HOLD = 14   # frames per case
CASE_COLORS = {"baseline": "C0", "peak_priority": "C1", "delay_priority": "C2",
               "fairness_priority": "C3"}


def main(scenario: str = "morning_peak", n_evs: int = 30, n_stations: int = 3, seed: int = 0,
         grid_factor=None):
    data = instance(scenario, n_evs, n_stations, seed, grid_factor)
    sols = {case: run_algorithm("centralized", data, w) for case, w in WEIGHT_CASES.items()}
    order = np.argsort(data.arrival, kind="stable")
    T, x = data.T, np.arange(data.T)
    vmax = max(r.P.max() for r in sols.values())

    fig, (ax, axh) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax.plot(x, data.grid_cap, "k--", lw=1, label="grid capacity")
    lines = {}
    for case, r in sols.items():
        lines[case] = ax.step(x, r.P.sum(0), where="post", color=CASE_COLORS[case],
                              lw=1, alpha=0.25, label=case.replace("_", " "))[0]
    ax.set_ylabel("total load [kW]")
    ax.set_xlim(0, T - 1)
    clock_ticks(ax, T, data.dt, data.start_hour)
    ax.legend(fontsize=8, loc="upper right")
    info = ax.text(0.01, 0.97, "", transform=ax.transAxes, va="top", family="monospace",
                   fontsize=8, bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.8"))
    im = axh.imshow(np.zeros((data.N, T)), aspect="auto", cmap="viridis", vmin=0, vmax=vmax,
                    interpolation="nearest")
    axh.set_ylabel("EV (sorted by arrival)")
    clock_ticks(axh, T, data.dt, data.start_hour)
    fig.colorbar(im, ax=axh, label="power [kW]")
    title = fig.suptitle("")
    cases = list(sols)

    def update(f):
        case = cases[min(f // HOLD, len(cases) - 1)]
        w = WEIGHT_CASES[case]
        r = sols[case]
        for c, ln in lines.items():
            ln.set_alpha(1.0 if c == case else 0.2)
            ln.set_linewidth(2.6 if c == case else 1.0)
        im.set_data(r.P[order])
        m = r.metrics
        info.set_text(f"peak   {m['peak_load_kw']:6.1f} kW\ncost   Rs {m['total_cost_inr']:7.0f}\n"
                      f"delay  {m['avg_delay_h']:5.2f} h\nfair   {m['jain_fairness']:.3f}")
        title.set_text(f"{case.replace('_', ' ')}   (alpha={w.alpha:g}, beta={w.beta:g}, gamma={w.gamma:g})")

    n_frames = HOLD * len(cases)
    update(0)
    tag = "" if grid_factor is None else f"_grid{grid_factor:g}"
    fig.savefig(png_path(f"weight_cases{tag}"), dpi=150)
    save_gif(fig, update, n_frames, f"weight_cases{tag}", fps=7, hold_last=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-factor", type=float, default=None)
    a = p.parse_args()
    main(a.scenario, a.n_evs, a.stations, a.seed, a.grid_factor)