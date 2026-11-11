"""
Animation: every EV's battery (SOC bar) through the day,
uncoordinated (top) vs coordinated Shared ADMM (bottom).

Bar colours: light grey = not arrived, green = charging, blue = plugged but
waiting, dark grey = departed.  Black tick = target SOC.

Run:  python -m simulations.sim_ev_soc
Out:  plots/animations/ev_soc_<scenario>.gif / .png
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from common.constraints import soc_trajectory
from common.objective import WEIGHT_CASES
from network.integration import run_algorithm
from simulations.sim_utils import instance, png_path, save_gif


def main(scenario: str = "morning_peak", n_evs: int = 30, n_stations: int = 3, seed: int = 0):
    data = instance(scenario, n_evs, n_stations, seed)
    w = WEIGHT_CASES["baseline"]
    asap = run_algorithm("uncoordinated_asap", data, w)
    coord = run_algorithm("shared_admm_adaptive", data, w, topology="star")
    runs = [("Uncoordinated (charge on arrival)", asap.P), ("Coordinated (Shared ADMM)", coord.P)]
    socs = [soc_trajectory(data, P) for _, P in runs]

    order = np.argsort(data.arrival, kind="stable")
    idx = np.arange(data.N)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.8), sharex=True)
    bars = []
    for ax, (title, P), soc in zip(axes, runs, socs):
        b = ax.bar(idx, soc[order, 0], color="0.85", edgecolor="0.5", lw=0.3)
        ax.scatter(idx, data.soc_target[order], marker="_", s=160, color="k", zorder=3,
                   label="target SOC")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("SOC")
        ax.set_title(title, fontsize=10)
        bars.append(b)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("EV (sorted by arrival time)")
    header = fig.suptitle("")

    def colour(i, t, P):
        if t < data.arrival[i]:
            return "0.88"
        if t >= data.departure[i]:
            return "0.45"
        return "#2ca02c" if P[i, t] > 0.05 else "#5b8fd1"

    def update(t):
        for b, (_, P), soc in zip(bars, runs, socs):
            for k, i in enumerate(order):
                b[k].set_height(soc[i, t + 1])
                b[k].set_color(colour(i, t, P))
        clock = (data.start_hour + t * data.dt) % 24
        loads = " | ".join(f"{name.split(' (')[0]}: {P[:, t].sum():5.1f} kW" for name, P in runs)
        header.set_text(f"{int(clock):02d}:{int(round((clock % 1) * 60)):02d}   total load  {loads}"
                        f"   (grid cap {data.grid_cap[t]:.0f} kW)")

    t_snap = int(np.argmax(asap.P.sum(0)))          # moment of the uncoordinated peak
    update(t_snap)
    fig.savefig(png_path(f"ev_soc_{scenario}"), dpi=150)
    save_gif(fig, update, data.T, f"ev_soc_{scenario}", fps=6)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.scenario, a.n_evs, a.stations, a.seed)