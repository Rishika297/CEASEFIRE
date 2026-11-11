"""
SOC and charging trajectory plots for two EV-level behaviours:
  * ASAP     : charge at full power on arrival (uncoordinated)
  * selfish  : each EV minimises its own cost+delay+fairness, ignoring others

Both ignore station/grid limits, which is exactly why coordination is needed:
the plot draws the grid capacity line for comparison.

Run:  python -m ev.experiments.ev_trajectories
Out:  plots/charging_schedules/ev_trajectories_<scenario>.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.constraints import soc_trajectory
from common.objective import WEIGHT_CASES, linear_coefficients
from common.utils import plots_dir
from ev.fleet import scenario_instance
from ev.local_opt import selfish_schedule
from ev.schedule import step_to_clock


def run(scenario: str = "evening_peak", n_evs: int = 50, n_stations: int = 3,
        seed: int = 0, n_show: int = 6) -> None:
    data, evs = scenario_instance(scenario, n_evs, n_stations, seed)
    w = WEIGHT_CASES["baseline"]

    P_asap = np.vstack([ev.asap_schedule(data.T, data.dt) for ev in evs])
    P_self, _ = selfish_schedule(data, linear_coefficients(data, w), w.gamma)

    hours = np.arange(data.T + 1)
    ticks = hours[::8]
    labels = [f"{h:02.0f}:00" for h in step_to_clock(ticks, data.dt, data.start_hour)]
    show = np.random.default_rng(seed).choice(data.N, n_show, replace=False)

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for name, P, ls in (("ASAP", P_asap, "-"), ("selfish", P_self, "--")):
        soc = soc_trajectory(data, P)
        for j, i in enumerate(show):
            ax[0].plot(hours, soc[i], ls, color=f"C{j}", label=f"EV {i} {name}" if name == "ASAP" else None)
        ax[1].step(np.arange(data.T), P.sum(axis=0), ls, where="post", label=name)
    for j, i in enumerate(show):
        ax[0].axhline(data.soc_target[i], color=f"C{j}", lw=0.6, alpha=0.5)
    ax[0].set_ylabel("SOC"); ax[0].set_title(f"SOC trajectories ({scenario}); solid=ASAP, dashed=selfish")
    ax[0].legend(fontsize=7, ncol=3)

    ax[1].plot(np.arange(data.T), data.grid_cap, "k:", label="grid capacity")
    ax[1].set_ylabel("total load [kW]"); ax[1].legend()

    ax[2].step(np.arange(data.T), data.prices, where="post", color="gray")
    ax[2].set_ylabel("price [Rs/kWh]")
    ax[2].set_xticks(ticks, labels)
    ax[2].set_xlabel("clock time")
    fig.tight_layout()

    out = plots_dir("charging_schedules") / f"ev_trajectories_{scenario}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"peak ASAP = {P_asap.sum(0).max():.1f} kW, peak selfish = {P_self.sum(0).max():.1f} kW, "
          f"grid cap = {data.grid_cap.max():.1f} kW")
    print(f"saved -> {out}")


if __name__ == "__main__":
    for sc in ("morning_peak", "evening_peak", "random"):
        run(sc)
