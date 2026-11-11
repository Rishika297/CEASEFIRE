"""
Story simulation with EVs as moving dots, side by side for
Uncoordinated | Centralized | Shared ADMM (adaptive) | Consensus ADMM (adaptive).

    Phase 1  Requests     EVs drive in one by one and announce their needs
    Phase 2  Negotiation  messages flow along the communication links
                          (to the operator / aggregator, or between neighbours);
                          each EV's colour = how far its plan still is from the
                          final agreement (red = far, green = agreed)
    Phase 3  The day      clock runs: EVs drive in at their arrival time, charge
                          (dot fills red -> green as its need is met, green ring =
                          charging, blue ring = waiting) and drive out.
Bottom row: total grid load of each approach vs the grid limit (red = overload).

Example: 12 EVs, 3 charging stations (4 chargers each), morning-peak arrivals.

Run:  python -m simulations.sim_story
      python -m simulations.sim_story --scenario evening_peak
      python -m simulations.sim_story --n-evs 16 --stations 4
Out:  plots/animations/story_<scenario>.gif
      plots/animations/story_<scenario>_phase1.png / _phase2.png / _phase3.png
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from common.constraints import soc_trajectory
from common.objective import WEIGHT_CASES
from evaluation.metrics import compute_metrics
from network.integration import make_network, run_algorithm
from simulations.sim_utils import COLORS, instance, iterates, png_path, save_gif

APPROACHES = ("uncoordinated", "centralized", "shared", "consensus")
TITLES = {"uncoordinated": "Uncoordinated", "centralized": "Centralized",
          "shared": "Shared ADMM (adaptive rho)", "consensus": "Consensus ADMM (adaptive rho)"}
EXPLAIN = {
    "uncoordinated": "No communication.\nEvery EV charges at full power\nas soon as it plugs in.",
    "centralized": "One operator collects every\nrequest and computes the\nschedule for everyone.",
    "shared": "EVs send their plans to an\naggregator and adjust them\nuntil the grid limit is shared.",
    "consensus": "No coordinator: each EV only\ntalks to its neighbours until\nall of them agree.",
}
CMAP = plt.get_cmap("RdYlGn")
NEG_FRAMES = 40          # frames for the negotiation phase
SUB = 2                  # animation sub-frames per 30-min interval in phase 3
HOLD = 8
ENTRANCE = np.array([-0.03, 0.10])
EXIT = np.array([1.03, 0.10])
HUB = np.array([0.50, 0.94])


def clock(data, t) -> str:
    h = (data.start_hour + t * data.dt) % 24
    return f"{int(h):02d}:{int(round((h % 1) * 60)):02d}"


def geometry(data):
    """Station boxes and one charger slot per EV."""
    M = data.M
    counts = np.bincount(data.station, minlength=M)
    K = int(counts.max())
    ncol = int(math.ceil(math.sqrt(K)))
    nrow = int(math.ceil(K / ncol))
    gap = 0.03
    bw = (0.94 - gap * (M - 1)) / M
    boxes, slot = [], np.zeros((data.N, 2))
    for m in range(M):
        x0 = 0.03 + m * (bw + gap)
        boxes.append((x0, 0.30, bw, 0.52))
        idx = np.flatnonzero(data.station == m)
        idx = idx[np.argsort(data.arrival[idx], kind="stable")]
        for k, i in enumerate(idx):
            r, c = divmod(k, ncol)
            slot[i] = (x0 + bw * (c + 0.5) / ncol, 0.30 + 0.46 * (1 - (r + 0.5) / nrow))
    return boxes, slot


def main(n_evs: int = 12, n_stations: int = 3, scenario: str = "morning_peak", seed: int = 0,
         grid_factor=None, fps: int = 8):
    data = instance(scenario, n_evs, n_stations, seed, grid_factor)
    N, T = data.N, data.T
    w = WEIGHT_CASES["baseline"]

    print("solving uncoordinated / centralized / shared ADMM / consensus ADMM ...", flush=True)
    res = {"uncoordinated": run_algorithm("uncoordinated_asap", data, w),
           "centralized": run_algorithm("centralized", data, w)}
    J_star = res["centralized"].objective
    res["shared"] = run_algorithm("shared_admm_adaptive", data, w, topology="star", J_star=J_star,
                                  store_iterates=True, store_every=1)
    g = make_network("random", N, seed=seed)
    res["consensus"] = run_algorithm("consensus_admm_adaptive", data, w, graph=g, J_star=J_star,
                                     store_iterates=True, store_every=5)
    its = {k: iterates(res[k]) for k in ("shared", "consensus")}
    every = {"shared": 1, "consensus": 5}
    est = np.asarray(res["consensus"].meta.get("estimate_iterates", []))

    P = {k: res[k].P for k in APPROACHES}
    soc = {k: soc_trajectory(data, P[k]) for k in APPROACHES}
    need = np.maximum(data.soc_target - data.soc_init, 1e-9)
    L_star = P["centralized"].sum(0)
    t_star = int(np.argmax(L_star))
    k_eta = data.eta * data.dt
    E_final = {k: k_eta * P[k].sum(1) for k in ("shared", "consensus")}
    boxes, slot = geometry(data)
    order = np.argsort(data.arrival, kind="stable")
    gcap = float(data.grid_cap.max())
    ymax = 1.15 * max(P["uncoordinated"].sum(0).max(), gcap)
    x_t = np.arange(T)

    fig, axes = plt.subplots(2, 4, figsize=(17, 7.6), gridspec_kw={"height_ratios": [1.55, 1]})
    scenes, panels = axes[0], axes[1]
    color_of = {"uncoordinated": COLORS["uncoordinated"], "centralized": COLORS["centralized"],
                "shared": COLORS["shared"], "consensus": COLORS["consensus"]}

    # ------------------------------------------------------------------ drawing
    def scene(ax, key, subtitle=""):
        ax.cla()
        ax.set_axis_off()
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(0.0, 1.02)
        ax.add_patch(Rectangle((-0.06, 0.04), 1.12, 0.12, color="0.85", zorder=0))
        ax.plot([-0.06, 1.06], [0.10, 0.10], color="white", lw=1.5, ls=(0, (6, 6)), zorder=1)
        ax.text(-0.04, 0.005, "in", fontsize=7, color="0.4")
        ax.text(0.98, 0.005, "out", fontsize=7, color="0.4")
        for m, (x0, y0, bw, bh) in enumerate(boxes):
            ax.add_patch(FancyBboxPatch((x0, y0), bw, bh, boxstyle="round,pad=0.01",
                                        fc="0.96", ec="0.6", lw=1, zorder=0))
            ax.text(x0 + bw / 2, y0 + bh + 0.012, f"Station {m + 1}  ({data.station_cap[m]:.0f} kW)",
                    ha="center", fontsize=6.5, weight="bold")
        for i in range(N):                                 # empty charger sockets
            ax.scatter(*slot[i], s=150, marker="s", c="white", edgecolors="0.75", lw=0.8, zorder=1)
        ax.set_title(TITLES[key] + (f"\n{subtitle}" if subtitle else ""), fontsize=9,
                     color=color_of[key], weight="bold")
        if key in ("centralized", "shared"):
            ax.scatter(*HUB, s=260, marker="s", c=[color_of[key]], zorder=4)
            ax.text(HUB[0] + 0.04, HUB[1], "operator" if key == "centralized" else "aggregator",
                    va="center", fontsize=7)

    def dot(ax, xy, face, edge="0.25", lw=1.0, alpha=1.0, ls="-", num=None, label=None):
        ax.scatter([xy[0]], [xy[1]], s=230, c=[face], edgecolors=[edge], linewidths=lw,
                   alpha=alpha, linestyle=ls, zorder=5)
        if num is not None:
            ax.text(xy[0], xy[1], str(num), ha="center", va="center", fontsize=6, zorder=6, alpha=alpha)
        if label:
            ax.text(xy[0], xy[1] - 0.055, label, ha="center", va="top", fontsize=5.5, zorder=6, alpha=alpha)

    def links(ax, key, present, phase, strong=True):
        """Communication links + travelling message dots between present agents."""
        if key == "uncoordinated" or len(present) == 0:
            return
        if key in ("centralized", "shared"):
            pairs = [(slot[i], HUB) for i in present]
        else:
            ps = set(present.tolist())
            pairs = [(slot[a], slot[b]) for a, b in g.edges if a in ps and b in ps]
        col = color_of[key]
        for p, q in pairs:
            ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=0.7, alpha=0.45 if strong else 0.15, zorder=2)
            if strong:
                for ph in (phase, (phase + 0.5) % 1.0):
                    m = p + (q - p) * ph
                    ax.scatter([m[0]], [m[1]], s=10, c=[col], zorder=3)

    def load_panel(ax, key, L, upto=None, ref=None, title=""):
        ax.cla()
        n = T if upto is None else upto + 1
        if ref is not None:
            ax.step(x_t, ref, where="post", color=COLORS["centralized"], lw=2.5, alpha=0.3,
                    label="optimum")
        ax.step(x_t[:n], L[:n], where="post", color=color_of[key], lw=2)
        ax.plot(x_t, data.grid_cap, "k--", lw=1)
        over = np.where(L[:n] > data.grid_cap[:n] + 1e-6, L[:n], np.nan)
        ax.fill_between(x_t[:n], data.grid_cap[:n], over, step="post", color="red", alpha=0.35)
        ax.set_xlim(0, T - 1)
        ax.set_ylim(0, ymax)
        ticks = np.arange(0, T + 1, 12)
        ax.set_xticks(ticks, [clock(data, t) for t in ticks], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_title(title, fontsize=8)
        if upto is not None:
            ax.axvline(upto, color="0.5", lw=0.8)

    def text_panel(ax, msg):
        ax.cla()
        ax.set_axis_off()
        ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9, color="0.3", transform=ax.transAxes)

    # ------------------------------------------------------------------ phase 1: requests
    def phase1(j, u):
        for key, ax, pn in zip(APPROACHES, scenes, panels):
            scene(ax, key)
            for r in range(j):
                i = order[r]
                dot(ax, slot[i], "white", edge="0.35", ls="--", num=i + 1, label=f"{data.R[i]:.0f} kWh")
            i = order[j]
            xy = ENTRANCE + (slot[i] - ENTRANCE) * u
            dot(ax, xy, "white", edge="0.35", ls="--", num=i + 1,
                label=f"{data.R[i]:.0f} kWh" if u >= 1 else None)
            text_panel(pn, EXPLAIN[key])
        i = order[j]
        fig.suptitle(f"1. EVs arrive with different needs - EV {i + 1}: needs {data.R[i]:.0f} kWh, "
                     f"here {clock(data, data.arrival[i])}-{clock(data, data.departure[i])}, "
                     f"charger {data.p_max[i]:.1f} kW", fontsize=11)

    # ------------------------------------------------------------------ phase 2: negotiation
    idx = {}
    for k in ("shared", "consensus"):
        n = len(its[k])
        idx[k] = np.unique(np.round(np.geomspace(1, n, NEG_FRAMES)).astype(int)) - 1

    def neg_index(k, f):
        arr = idx[k]
        return int(arr[min(int(f * len(arr) / NEG_FRAMES), len(arr) - 1)])

    def phase2(f):
        u = f / max(NEG_FRAMES - 1, 1)
        phase = (f * 0.25) % 1.0
        allv = np.arange(N)
        spread_txt = ""
        for key, ax, pn in zip(APPROACHES, scenes, panels):
            if key == "uncoordinated":
                scene(ax, key, "no negotiation")
                for i in allv:
                    dot(ax, slot[i], CMAP(0.0), num=i + 1, label=f"wants {data.p_max[i]:.0f} kW now")
                L = P[key].sum(0)
                load_panel(pn, key, L, title=f"plans add up to {L.max():.0f} kW  (limit {gcap:.0f} kW)")
            elif key == "centralized":
                decided = u >= 0.5
                scene(ax, key, "decision sent to all EVs" if decided else "collecting requests")
                links(ax, key, allv, phase if not decided else 1 - phase)
                for i in allv:
                    dot(ax, slot[i], CMAP(1.0) if decided else "0.85", num=i + 1,
                        label="schedule received" if decided else "sends request")
                if decided:
                    load_panel(pn, key, L_star, title=f"one optimisation: peak {L_star.max():.0f} kW")
                else:
                    text_panel(pn, "operator is solving ...")
            else:
                k = neg_index(key, f)
                Pk = its[key][k]
                Ek = k_eta * Pk.sum(1)
                dev = np.abs(Ek - E_final[key]) / np.maximum(data.R, 1.0)
                h = res[key].history
                hk = min(k * every[key], len(h["gap"]) - 1)
                it_no = int(h["iteration"][hk])
                scene(ax, key, f"negotiation round {it_no}")
                links(ax, key, allv, phase)
                for i in allv:
                    if key == "consensus" and est.size:
                        lab = f"thinks {est[min(k, len(est) - 1), i, t_star]:.0f} kW"
                    else:
                        lab = f"plans {Ek[i]:.0f} kWh"
                    dot(ax, slot[i], CMAP(1.0 - min(3 * dev[i], 1.0)), num=i + 1, label=lab)
                load_panel(pn, key, Pk.sum(0), ref=L_star,
                           title=f"round {it_no}: gap {h['gap'][hk]:.0e}, over limit "
                                 f"{h['max_violation'][hk]:.1f} kW")
                if key == "consensus" and est.size:
                    e = est[min(k, len(est) - 1), :, t_star]
                    spread_txt = f"   (consensus EVs disagree by {e.max() - e.min():.1f} kW)"
        fig.suptitle("2. Different opinions -> one agreement (red = far from agreement, green = agreed)"
                     + spread_txt, fontsize=11)

    # ------------------------------------------------------------------ phase 3: the day
    def phase3(t, u):
        for key, ax, pn in zip(APPROACHES, scenes, panels):
            Pk, sk = P[key], soc[key]
            present = np.flatnonzero((data.arrival <= t) & (data.departure > t))
            ok = short = 0
            scene(ax, key)
            links(ax, key, present, (t * SUB + u * SUB) * 0.25 % 1.0, strong=False)
            for i in range(N):
                a, d = int(data.arrival[i]), int(data.departure[i])
                soc_now = sk[i, t] + (sk[i, t + 1] - sk[i, t]) * u
                frac = float(np.clip((soc_now - data.soc_init[i]) / need[i], 0, 1))
                if data.R[i] <= 1e-9:
                    frac = 1.0
                if t < a - 1:
                    dot(ax, slot[i], "white", edge="0.6", ls="--", alpha=0.35, num=i + 1,
                        label=clock(data, a))
                elif t == a - 1:
                    dot(ax, ENTRANCE + (slot[i] - ENTRANCE) * u, CMAP(frac), num=i + 1)
                elif a <= t < d:
                    on = Pk[i, t] > 0.05
                    dot(ax, slot[i], CMAP(frac), edge="#00c000" if on else "#3b78d8",
                        lw=3 if on else 2, num=i + 1,
                        label=f"{frac * 100:.0f}%" + (f" {Pk[i, t]:.0f}kW" if on else ""))
                elif t == d:
                    dot(ax, slot[i] + (EXIT - slot[i]) * u, CMAP(frac), num=i + 1)
                else:
                    f_end = float(np.clip((sk[i, d] - data.soc_init[i]) / need[i], 0, 1)) \
                        if data.R[i] > 1e-9 else 1.0
                    ok += f_end >= 0.95
                    short += f_end < 0.95
            ax.text(0.0, 0.2, f"left with need met: {ok}   short: {short}", fontsize=7, color="0.3")
            L = Pk.sum(0)
            cost = float(data.prices[:t + 1] @ L[:t + 1] * data.dt)
            load_panel(pn, key, L, upto=t,
                       title=f"now {L[t]:.0f} kW   peak {L[:t + 1].max():.0f} kW   cost Rs {cost:.0f}")
        fig.suptitle(f"3. Charging and departure - {clock(data, t)}    "
                     f"(fill: red -> green as the need is met | ring: green = charging, blue = waiting)",
                     fontsize=11)

    # ------------------------------------------------------------------ schedule
    sched = []
    for j in range(N):
        sched += [("p1", j, 0.5), ("p1", j, 1.0)]
    sched += [("p1", N - 1, 1.0)] * HOLD
    sched += [("p2", f, 0.0) for f in range(NEG_FRAMES)] + [("p2", NEG_FRAMES - 1, 0.0)] * HOLD
    t0 = max(int(data.arrival.min()) - 1, 0)
    t1 = min(int(data.departure.max()) + 1, T - 1)
    sched += [("p3", t, s / SUB) for t in range(t0, t1 + 1) for s in range(SUB)]
    sched += [("p3", t1, 0.99)] * (2 * HOLD)

    def render(fr):
        kind, a, b = sched[fr]
        if kind == "p1":
            phase1(a, b)
        elif kind == "p2":
            phase2(a)
        else:
            phase3(a, b)
        fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.06, hspace=0.18, wspace=0.12)

    tag = f"story_{scenario}"
    render(2 * N - 1)
    fig.savefig(png_path(f"{tag}_phase1"), dpi=150)
    render(sched.index(("p2", NEG_FRAMES - 1, 0.0)))
    fig.savefig(png_path(f"{tag}_phase2"), dpi=150)
    t_peak = int(np.argmax(P["uncoordinated"].sum(0)))
    phase3(t_peak, 0.0)
    fig.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.06, hspace=0.18, wspace=0.12)
    fig.savefig(png_path(f"{tag}_phase3"), dpi=150)

    print(f"\n{N} EVs, {data.M} stations, grid limit {gcap:.0f} kW, scenario {scenario}")
    for key in APPROACHES:
        r = res[key]
        m = compute_metrics(data, r.P, None, w)
        extra = f"  rounds={r.iterations} messages={r.messages}" if key in ("shared", "consensus") else ""
        print(f"{TITLES[key]:30s} peak={m['peak_load_kw']:6.1f} kW  cost=Rs {m['total_cost_inr']:7.0f}  "
              f"need met={m['mean_satisfaction'] * 100:5.1f}%  over limit={m['max_violation']:5.1f} kW{extra}")
    if est.size:
        e0, e1 = est[0, :, t_star], est[-1, :, t_star]
        print(f"consensus opinions at {clock(data, t_star)}: spread {e0.max() - e0.min():.1f} kW "
              f"-> {e1.max() - e1.min():.2f} kW")

    save_gif(fig, render, len(sched), tag, fps=fps, hold_last=0)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Story simulation: arrive, negotiate, charge, leave")
    p.add_argument("--n-evs", type=int, default=12)
    p.add_argument("--stations", type=int, default=3)
    p.add_argument("--scenario", default="morning_peak")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-factor", type=float, default=None)
    p.add_argument("--fps", type=int, default=8)
    a = p.parse_args()
    main(a.n_evs, a.stations, a.scenario, a.seed, a.grid_factor, a.fps)
    