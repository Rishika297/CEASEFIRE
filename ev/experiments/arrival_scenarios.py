"""
Arrival and departure scenarios (Experiment 5 inputs).

For each scenario plots the arrival histogram and the number of plugged-in
EVs over the day, and prints a summary table.

Run:  python -m ev.experiments.arrival_scenarios
Out:  plots/scenarios/arrival_scenarios.png, results/scenarios/arrival_summary.csv
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.constants import EXP5_SCENARIOS
from common.utils import plots_dir, results_dir
from ev.fleet import fleet_summary, scenario_instance
from ev.schedule import occupancy, step_to_clock


def run(n_evs: int = 100, n_stations: int = 5, seed: int = 0) -> pd.DataFrame:
    names = list(EXP5_SCENARIOS)
    fig, ax = plt.subplots(2, len(names), figsize=(4 * len(names), 6), sharey="row")
    rows = []
    for j, name in enumerate(names):
        data, evs = scenario_instance(name, n_evs, n_stations, seed)
        occ = occupancy(data.arrival, data.departure, data.T)
        ax[0, j].hist(data.arrival, bins=np.arange(data.T + 1), color="C0", alpha=0.7, label="arrival")
        ax[0, j].hist(data.departure - 1, bins=np.arange(data.T + 1), color="C3", alpha=0.5, label="departure")
        ax[0, j].set_title(name)
        ax[1, j].step(np.arange(data.T), occ, where="post")
        ticks = np.arange(0, data.T + 1, 8)
        ax[1, j].set_xticks(ticks, [f"{h:02.0f}" for h in step_to_clock(ticks, data.dt, data.start_hour)])
        ax[1, j].set_xlabel("clock hour")
        row = fleet_summary(evs, data.dt)
        row.update({"scenario": name, "peak_occupancy": int(occ.max()),
                    "mean_arrival_clock_h": float(np.mean(step_to_clock(data.arrival, data.dt, data.start_hour)))})
        rows.append(row)
    ax[0, 0].set_ylabel("# EVs"); ax[0, 0].legend()
    ax[1, 0].set_ylabel("# plugged in")
    fig.tight_layout()
    out = plots_dir("scenarios") / "arrival_scenarios.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    df = pd.DataFrame(rows).set_index("scenario")
    df.to_csv(results_dir("scenarios") / "arrival_summary.csv")
    print(df.round(2))
    print(f"saved -> {out}")
    return df


if __name__ == "__main__":
    run()
