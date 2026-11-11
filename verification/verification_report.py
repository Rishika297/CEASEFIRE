"""
Text reports and plots for STL verification results.

    print(format_report(rep))
    plot_soc_verification(data, P, rep, "plots/verification/soc.png")
    plot_grid_verification(data, P, rep, "plots/verification/grid.png")
    plot_stl_comparison(df, "plots/verification/comparison.png")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from common.constraints import soc_trajectory


def _hhmm(h: Optional[float]) -> str:
    if h is None:
        return "  -  "
    h = h % 24.0
    return f"{int(h):02d}:{int(round((h % 1) * 60)) % 60:02d}"


def _pp(x: float) -> str:
    """SoC robustness in percentage points, without '-0.00'."""
    return f"{round(100 * x, 2) + 0.0:+.2f}"


def _ok(flag: bool) -> str:
    return "SATISFIED" if flag else "VIOLATED"


# ======================================================================
def format_report(rep, max_evs: Optional[int] = 15) -> str:
    """Human-readable STL verification report."""
    L = []
    title = "STL VERIFICATION REPORT"
    if rep.algorithm:
        title += f"  -  {rep.algorithm}"
    if rep.scenario:
        title += f"  [{rep.scenario}]"
    L += [title, "=" * len(title), ""]
    L += ["Specifications",
          "  Req 1 (per EV):  F_[a_i, d_i] (SoC_i >= SoC_i^target)   target SoC before departure",
          "  Req 2 (grid):    G_[0, T-1]  (P_total <= P_grid)         grid capacity never exceeded",
          "  Robustness: Req 1 in SoC percentage points, Req 2 in kW (positive = satisfied)", ""]

    # ---------------- Requirement 1 ----------------
    n, viol = len(rep.evs), rep.violated_evs
    L.append(f"REQUIREMENT 1 - target SoC before departure: {_ok(rep.req1_satisfied)}")
    L.append(f"  EVs satisfied: {n - len(viol)}/{n}"
             f"   min robustness: {_pp(rep.rho_req1)} pp")
    row = rep.to_row()
    if row["stl_req1_min_margin_h"] == row["stl_req1_min_margin_h"]:     # not NaN
        L.append(f"  deadline margin (target reached before departure): "
                 f"min {row['stl_req1_min_margin_h']:.1f} h, "
                 f"mean {row['stl_req1_mean_margin_h']:.1f} h")
    if viol:
        unreach = sum(not e.reachable for e in viol)
        L.append(f"  violated: {len(viol)}  (of which {unreach} cannot reach target even "
                 f"charging alone at P_max for the whole stay)")
    L.append("")
    hdr = (f"  {'EV':>4} {'stn':>3} {'arrive':>6} {'depart':>6} {'SoC0':>6} "
           f"{'target':>6} {'SoC@dep':>7} {'rho[pp]':>8} {'reached':>7} {'margin':>7}  status")
    L += [hdr, "  " + "-" * (len(hdr) - 2)]
    # worst first: violations by robustness, then satisfied EVs by smallest deadline margin
    evs = sorted(rep.evs, key=lambda e: (e.satisfied, e.rho if not e.satisfied else 0.0,
                                         e.deadline_margin_h if e.deadline_margin_h is not None else 0.0))
    shown = evs if max_evs is None else evs[:max_evs]
    for e in shown:
        margin = "-" if e.deadline_margin_h is None else f"{e.deadline_margin_h:.1f}h"
        L.append(f"  {e.ev_id:>4} {e.station:>3} {_hhmm(e.arrival_clock):>6} "
                 f"{_hhmm(e.departure_clock):>6} {100 * e.soc_init:5.1f}% "
                 f"{100 * e.soc_target:5.1f}% {100 * e.soc_at_departure:6.1f}% "
                 f"{_pp(e.rho):>8} {_hhmm(e.reached_clock):>7} {margin:>7}  {e.status}")
    if max_evs is not None and len(evs) > max_evs:
        L.append(f"  ... {len(evs) - max_evs} more EVs (sorted worst first; "
                 f"use ev_table() for the full list)")
    L.append("")

    # ---------------- Requirement 2 ----------------
    g = rep.grid
    L.append(f"REQUIREMENT 2 - grid capacity: {_ok(g.satisfied)}")
    L.append(f"  Maximum capacity: {g.cap_kw:.1f} kW")
    L.append(f"  Observed peak:    {g.peak_kw:.1f} kW at {_hhmm(rep.start_hour + g.t_peak * rep.dt)}")
    L.append(f"  Robustness:       {g.rho_kw:+.2f} kW  ({100 * g.rho_norm:+.1f} % of capacity), "
             f"critical time {_hhmm(rep.start_hour + g.t_critical * rep.dt)}")
    if g.violations:
        L.append(f"  Violations: {len(g.violations)} episode(s), "
                 f"{g.n_violation_steps * rep.dt:.1f} h over the limit in total")
        for v in g.violations:
            who = ", ".join(f"EV{c['ev_id']} ({c['kw']:.1f} kW)" for c in v.top_contributors)
            L.append(f"    {_hhmm(v.start_clock)} for {v.duration_h:.1f} h: worst excess "
                     f"{v.worst_excess_kw:.1f} kW, overload {v.overload_kwh:.1f} kWh; "
                     f"largest loads: {who}")
    L.append("")

    # ---------------- Overall ----------------
    L.append(f"Overall STL status: {_ok(rep.satisfied)}   "
             f"(overall robustness {rep.rho_overall:+.4f}, normalised)")
    if not rep.satisfied:
        why = []
        if not rep.req1_satisfied:
            unreach = sum(not e.reachable for e in viol)
            why.append(f"{len(viol)} EV(s) miss their target SoC"
                       + (f" ({unreach} physically unreachable)" if unreach else ""))
        if not rep.req2_satisfied:
            why.append("grid capacity exceeded")
        L.append("  because: " + "; ".join(why))
        r = rep.to_row()
        L.append(f"Restricted to EVs with a reachable target: "
                 f"{_ok(r['stl_satisfied_reachable'])} "
                 f"({r['stl_req1_reachable_sat_pct']:.1f} % of reachable EVs meet Req 1)")
    return "\n".join(L)


# ======================================================================
# Plots
# ======================================================================
def _clock_axis(ax, T, dt, start_hour, every_h=4):
    step = max(int(round(every_h / dt)), 1)
    ticks = np.arange(0, T + 1, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([_hhmm(start_hour + t * dt) for t in ticks], fontsize=7)


def plot_soc_verification(data, P, rep, path, n_show: int = 8):
    """SoC trajectories of the worst EVs with target and deadline marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    soc = soc_trajectory(data, P)
    idx_of = {int(e): i for i, e in enumerate(data.ev_ids)}
    chosen = sorted(rep.evs, key=lambda e: (e.satisfied, e.rho if not e.satisfied else 0.0,
                                            e.deadline_margin_h or 0.0))[:n_show]
    ncols = 4
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows),
                             squeeze=False, sharey=True)
    t = np.arange(data.T + 1)
    for ax, e in zip(axes.flat, chosen):
        i = idx_of[e.ev_id]
        col = "#2a9d8f" if e.satisfied else "#d1495b"
        ax.axvspan(e.arrival, e.departure, color="0.92", lw=0)
        ax.plot(t, 100 * soc[i], color=col, lw=1.8)
        ax.axhline(100 * e.soc_target, ls="--", color="0.3", lw=1)
        ax.axvline(e.departure, ls=":", color="0.3", lw=1)
        if e.t_reached is not None:
            ax.plot(e.t_reached, 100 * soc[i, e.t_reached], "o", color=col, ms=4)
        ax.set_title(f"EV{e.ev_id}  rho={_pp(e.rho)} pp  {e.status.split()[0]}",
                     fontsize=8, color=col)
        _clock_axis(ax, data.T, data.dt, data.start_hour)
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(chosen):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("SoC [%]")
    fig.suptitle(f"Req 1: F[a,d](SoC >= target) - {rep.algorithm}  "
                 "(grey = plugged in, dashed = target, dotted = departure)", fontsize=9)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_grid_verification(data, P, rep, path):
    """Aggregate load vs capacity, and the robustness signal P_grid - L_t."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = np.asarray(P).sum(axis=0)
    cap = np.asarray(data.grid_cap, dtype=float)
    margin = cap - L
    t = np.arange(data.T)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.step(t, L, where="post", color="#264653", lw=1.8, label="P_total")
    a1.step(t, cap, where="post", color="#d1495b", ls="--", lw=1.4, label="P_grid")
    for v in rep.grid.violations:
        a1.axvspan(v.start, v.end + 1, color="#d1495b", alpha=0.15, lw=0)
    a1.set_ylabel("power [kW]")
    a1.legend(fontsize=8, loc="upper left")
    a1.set_title(f"Req 2: G(P_total <= P_grid) - {rep.algorithm}: "
                 f"{'SATISFIED' if rep.grid.satisfied else 'VIOLATED'}", fontsize=10)
    a1.grid(alpha=0.3)

    a2.step(t, margin, where="post", color="#2a9d8f", lw=1.5)
    a2.fill_between(t, margin, 0, where=margin < 0, step="post", color="#d1495b", alpha=0.4)
    a2.axhline(0, color="0.2", lw=0.8)
    tc = rep.grid.t_critical
    a2.plot(tc, margin[tc], "v", color="0.1", ms=6)
    a2.annotate(f"rho = {rep.grid.rho_kw:+.1f} kW", (tc, margin[tc]),
                textcoords="offset points", xytext=(6, -12), fontsize=8)
    a2.set_ylabel("r(t) = P_grid - P_total")
    a2.grid(alpha=0.3)
    _clock_axis(a2, data.T, data.dt, data.start_hour)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_stl_comparison(df, path, group: str = "label"):
    """Bar charts of the STL metrics per algorithm (mean over rows of each group)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("stl_req1_reachable_sat_pct", "Req 1: reachable EVs satisfied [%]", 1.0),
              ("stl_req1_mean_margin_h", "Req 1: mean deadline margin [h] (worst scenario)", 1.0),
              ("stl_req2_rho_kw", "Req 2: robustness [kW] (worst scenario)", 1.0),
              ("stl_req2_viol_hours", "Req 2: hours over capacity", 1.0)]
    # robustness = worst case over scenarios/seeds (STL semantics); the rest = mean
    agg = {c: ("min" if "rho" in c or "margin" in c else "mean") for c, _, _ in panels}
    agg["stl_req2_viol_hours"] = "mean"
    g = df.groupby(group, sort=False).agg(agg)
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.6))
    for ax, (col, title, k) in zip(axes, panels):
        vals = g[col].to_numpy() * k
        colors = ["#2a9d8f" if (v >= 0 if "rho" in col else True) else "#d1495b" for v in vals]
        if col == "stl_req2_viol_hours":
            colors = ["#d1495b" if v > 0 else "#2a9d8f" for v in vals]
        ax.bar(range(len(vals)), vals, color=colors)
        ax.axhline(0, color="0.2", lw=0.8)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(g.index, rotation=35, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
