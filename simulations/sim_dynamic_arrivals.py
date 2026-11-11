"""
Animation: dynamic EV arrivals with online (rolling-horizon) distributed control.

Every interval t the EVs that are currently plugged in (and only those - future
arrivals are unknown) re-optimise the rest of the day with Shared ADMM, starting
from their current SOC. Only the first interval of the plan is applied.
Compared with: uncoordinated charging and the offline optimum (perfect knowledge
of all arrivals).

Run:  python -m simulations.sim_dynamic_arrivals                    (~1-2 min)
      python -m simulations.sim_dynamic_arrivals --scenario random
Out:  plots/animations/dynamic_<scenario>.gif / .png  + printed comparison table
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from common.objective import WEIGHT_CASES
from evaluation.metrics import compute_metrics
from network.integration import run_algorithm
from simulations.sim_utils import COLORS, clock_ticks, instance, png_path, save_gif


def rolling_horizon(data, w, algorithm="shared_admm_adaptive", topology="star", max_iter=3000):
    N, T = data.N, data.T
    soc = data.soc_init.astype(float).copy()
    gain = data.eta * data.dt / data.battery
    applied = np.zeros((N, T))
    plans, n_active, iters = [], [], []
    for t in range(T):
        active = np.flatnonzero((data.arrival <= t) & (data.departure > t))
        n_active.append(int(active.size))
        plan = np.zeros(T)
        if active.size:
            sub = data.subset(active)
            sub = replace(sub, soc_init=np.clip(soc[active], sub.soc_min, sub.soc_max))
            sub.avail[:, :t] = False                       # the past cannot be changed
            sub.p_upper = sub.p_max[:, None] * sub.avail
            r = run_algorithm(algorithm, sub, w, topology=topology, max_iter=max_iter)
            P = np.clip(r.P, 0.0, sub.p_upper)
            applied[active, t] = P[:, t]
            soc[active] = soc[active] + gain[active] * P[:, t]
            plan[t:] = P[:, t:].sum(axis=0)
            iters.append(r.iterations)
        plans.append(plan)
    return applied, plans, np.array(n_active), iters


def main(scenario: str = "morning_peak", n_evs: int = 30, n_stations: int = 3, seed: int = 0):
    data = instance(scenario, n_evs, n_stations, seed)
    w = WEIGHT_CASES["baseline"]
    asap = run_algorithm("uncoordinated_asap", data, w)
    offline = run_algorithm("centralized", data, w)
    print("running online rolling-horizon Shared ADMM ...", flush=True)
    applied, plans, n_active, iters = rolling_horizon(data, w)

    keys = ("peak_load_kw", "total_cost_inr", "avg_delay_h", "target_soc_pct",
            "total_unmet_kwh", "max_violation")
    print(f"\nre-optimisations: {len(iters)}, mean ADMM iterations each: {np.mean(iters):.0f}")
    for name, P in (("uncoordinated", asap.P), ("online ADMM", applied), ("offline optimum", offline.P)):
        m = compute_metrics(data, P, None, w)
        print(f"{name:16s} " + "  ".join(f"{k}={m[k]:.2f}" for k in keys))

    T, x = data.T, np.arange(data.T)
    L_off, L_asap = offline.P.sum(0), asap.P.sum(0)
    ymax = 1.15 * max(L_asap.max(), data.grid_cap.max())
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    ax.plot(x, data.grid_cap, "k--", lw=1, label="grid capacity")
    ax.step(x, L_asap, where="post", color=COLORS["uncoordinated"], lw=1, alpha=0.35,
            label="uncoordinated")
    ax.step(x, L_off, where="post", color=COLORS["centralized"], lw=1.5, alpha=0.45,
            label="offline optimum (knows the future)")
    done = ax.step([], [], where="post", color=COLORS["shared"], lw=2.6, label="online ADMM: applied")[0]
    plan_ln = ax.step([], [], where="post", color=COLORS["shared"], lw=1.4, ls="--",
                      label="online ADMM: current plan")[0]
    ax.set_ylim(0, ymax)
    ax.set_xlim(0, T - 1)
    ax.set_ylabel("total load [kW]")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title(f"Dynamic arrivals: EVs re-plan every {int(data.dt * 60)} min with only the EVs present")
    bars = ax2.bar(x, n_active, color="0.8", width=0.9)
    ax2.set_ylabel("EVs\nplugged in")
    clock_ticks(ax2, T, data.dt, data.start_hour)
    v = ax.axvline(0, color="k", lw=0.8, alpha=0.4)
    L_app = applied.sum(0)

    def update(t):
        done.set_data(x[:t + 1], L_app[:t + 1])
        plan_ln.set_data(x[t:], plans[t][t:])
        v.set_xdata([t, t])
        for k, b in enumerate(bars):
            b.set_color(COLORS["shared"] if k == t else ("0.55" if k < t else "0.85"))

    update(T - 1)
    fig.savefig(png_path(f"dynamic_{scenario}"), dpi=150)
    save_gif(fig, update, T, f"dynamic_{scenario}", fps=5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--n-evs", type=int, default=30)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.scenario, a.n_evs, a.stations, a.seed)