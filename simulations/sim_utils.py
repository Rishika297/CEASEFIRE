"""
Shared helpers for the presentation animations (GIFs via Pillow).
Outputs go to plots/animations/.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from common.utils import plots_dir
from ev.fleet import scenario_instance

COLORS = {
    "uncoordinated": "#d62728",
    "centralized": "#1f77b4",
    "shared": "#2ca02c",
    "consensus": "#9467bd",
}
TOPO_COLORS = {"ring": "C0", "star": "C1", "mesh": "C2", "random": "C3"}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def instance(scenario: str = "morning_peak", n_evs: int = 30, n_stations: int = 3,
             seed: int = 0, grid_factor=None):
    extra = {} if grid_factor is None else {"grid_cap_factor": grid_factor}
    data, _ = scenario_instance(scenario, n_evs, n_stations, seed, **extra)
    return data


def clock_ticks(ax, T: int, dt: float, start_hour: float, every_h: float = 4.0):
    step = max(1, int(round(every_h / dt)))
    ticks = np.arange(0, T + 1, step)
    ax.set_xticks(ticks, [f"{(start_hour + t * dt) % 24:02.0f}:00" for t in ticks])


def iterates(res):
    """Stored ADMM iterates (needs store_iterates=True) as a list of (N, T) arrays."""
    return [np.asarray(P) for P in res.meta.get("iterates", [])]


def png_path(name: str):
    return plots_dir("animations") / f"{name}.png"


def save_gif(fig, update, n_frames: int, name: str, fps: int = 8, hold_last: int = 12):
    """Render update(frame) for frame = 0..n_frames-1 (last frame held) to a GIF."""
    frames = list(range(n_frames)) + [n_frames - 1] * hold_last
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out = plots_dir("animations") / f"{name}.gif"
    anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved -> {out}")
    return out