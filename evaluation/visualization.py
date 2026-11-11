"""
Common plotting functions (matplotlib, saved as PNG).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TOPO_COLORS = {"ring": "C0", "star": "C1", "mesh": "C2", "random": "C3", "hub": "k"}


def _save(fig, out: Optional[Path]):
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
    plt.close(fig)


def _order(values) -> list:
    return list(dict.fromkeys(values))


def _clock_ticks(ax, T: int, dt: float, start_hour: float, every_h: float = 4.0):
    step = max(1, int(round(every_h / dt)))
    ticks = np.arange(0, T + 1, step)
    ax.set_xticks(ticks, [f"{(start_hour + t * dt) % 24:02.0f}" for t in ticks])


# ----------------------------------------------------------------- Experiment 1
def plot_convergence_grid(results: Dict[Tuple[str, str], object], key: str = "gap",
                          out: Optional[Path] = None, logy: bool = True,
                          ylabel: Optional[str] = None) -> None:
    """One panel per algorithm, one line per topology: history[key] vs iteration."""
    algos = sorted({a for a, _ in results})
    fig, axes = plt.subplots(1, len(algos), figsize=(4.3 * len(algos), 3.6),
                             sharey=True, squeeze=False)
    for ax, algo in zip(axes[0], algos):
        for (a, topo), r in sorted(results.items()):
            if a != algo or key not in r.history:
                continue
            y = np.asarray(r.history[key], dtype=float)
            if logy:
                y = np.maximum(y, 1e-12)
            ax.plot(r.history["iteration"], y, label=topo, color=TOPO_COLORS.get(topo), lw=1.2)
        ax.set_title(algo, fontsize=10)
        ax.set_xlabel("iteration")
        if logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel(ylabel or key)
    axes[0][-1].legend(fontsize=8)
    _save(fig, out)


def plot_metric_bars(df, metric: str, out: Optional[Path] = None, logy: bool = False,
                     ylabel: Optional[str] = None) -> None:
    """Grouped bars: x = algorithm, one bar per topology (mean over seeds)."""
    piv = df.groupby(["algorithm", "topology"])[metric].mean().unstack("topology")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    n = piv.shape[1]
    width = 0.8 / max(n, 1)
    x = np.arange(piv.shape[0])
    for j, topo in enumerate(piv.columns):
        ax.bar(x + j * width - 0.4 + width / 2, piv[topo].values, width,
               label=topo, color=TOPO_COLORS.get(topo))
    ax.set_xticks(x, piv.index, rotation=15, fontsize=9)
    ax.set_ylabel(ylabel or metric)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, out)


# ----------------------------------------------------------------- generic
def plot_metric_panels(df, index: str, columns: str, metrics: Sequence[str],
                       out: Optional[Path] = None, logy: Iterable[str] = (),
                       ncols: int = 3) -> None:
    """One panel per metric; grouped bars with x = `index`, one bar per `columns`
    value (mean over seeds)."""
    logy = set(logy)
    idx_order, col_order = _order(df[index]), _order(df[columns])
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows), squeeze=False)
    width = 0.8 / max(len(col_order), 1)
    x = np.arange(len(idx_order))
    for ax, metric in zip(axes.flat, metrics):
        piv = df.groupby([index, columns])[metric].mean().unstack(columns)
        piv = piv.reindex(index=idx_order, columns=col_order)
        for j, col in enumerate(col_order):
            ax.bar(x + j * width - 0.4 + width / 2, piv[col].values, width, label=col)
        ax.set_xticks(x, idx_order, rotation=20, fontsize=8)
        ax.set_title(metric, fontsize=10)
        if metric in logy:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    _save(fig, out)


def plot_lines(df, x: str, hue: str, metrics: Sequence[str], out: Optional[Path] = None,
               logy: Iterable[str] = (), ncols: int = 3) -> None:
    """One panel per metric; one line per `hue` value vs `x` (mean over seeds)."""
    logy = set(logy)
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax, metric in zip(axes.flat, metrics):
        piv = df.groupby([x, hue])[metric].mean().unstack(hue)
        for col in _order(df[hue]):
            if col in piv:
                ax.plot(piv.index, piv[col], "o-", label=col)
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel(x)
        if metric in logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(metrics):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    _save(fig, out)


def plot_load_grid(panels: Dict[str, Dict[str, np.ndarray]], grid_cap: np.ndarray,
                   out: Optional[Path] = None, dt: float = 0.5, start_hour: float = 6.0,
                   ncols: int = 3) -> None:
    """One panel per key of `panels`; each panel = {label: L_t} total-load curves."""
    names = list(panels)
    nrows = math.ceil(len(names) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.3 * nrows),
                             squeeze=False, sharey=True)
    T = len(grid_cap)
    for ax, name in zip(axes.flat, names):
        for lab, L in panels[name].items():
            ax.step(np.arange(T), L, where="post", label=lab, lw=1.2)
        ax.plot(np.arange(T), grid_cap, "k:", lw=1, label="grid capacity")
        ax.set_title(name, fontsize=10)
        _clock_ticks(ax, T, dt, start_hour)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    axes.flat[0].set_ylabel("total load [kW]")
    axes.flat[0].legend(fontsize=7)
    _save(fig, out)


def plot_schedule_heatmaps(schedules: Dict[str, np.ndarray], out: Optional[Path] = None,
                           dt: float = 0.5, start_hour: float = 6.0,
                           order: Optional[np.ndarray] = None, ncols: int = 2) -> None:
    """Heatmap of P (EV x time) for each schedule. `order` sorts the EV rows
    (e.g. by arrival)."""
    names = list(schedules)
    nrows = math.ceil(len(names) / ncols)
    vmax = max(float(np.max(P)) for P in schedules.values()) or 1.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.6 * nrows), squeeze=False)
    im = None
    for ax, name in zip(axes.flat, names):
        P = schedules[name] if order is None else schedules[name][order]
        im = ax.imshow(P, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_ylabel("EV (sorted by arrival)")
        _clock_ticks(ax, P.shape[1], dt, start_hour)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="power [kW]")
    if out is not None:
        fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_load_profiles(profiles: Dict[str, np.ndarray], grid_cap: np.ndarray,
                       out: Optional[Path] = None, dt: float = 0.5) -> None:
    """Total load vs time for several schedules (name -> L_t)."""
    fig, ax = plt.subplots(figsize=(8, 3.6))
    t = np.arange(len(grid_cap)) * dt
    for name, L in profiles.items():
        ax.step(t, L, where="post", label=name)
    ax.plot(t, grid_cap, "k:", label="grid capacity")
    ax.set_xlabel("hours from start")
    ax.set_ylabel("total load [kW]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, out)